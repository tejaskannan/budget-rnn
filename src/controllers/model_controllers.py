import numpy as np
import os.path
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from scipy.stats import norm
from typing import List, Optional, Tuple, Dict

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from utils.np_utils import index_of, round_to_precision
from utils.constants import OUTPUT, BIG_NUMBER, SMALL_NUMBER, INPUTS, SEQ_LENGTH, DROPOUT_KEEP_RATE, SEQ_LENGTH, NUM_CLASSES
from utils.file_utils import save_pickle_gz, read_pickle_gz, extract_model_name
from utils.loading_utils import restore_neural_network
from controllers.power_distribution import PowerDistribution
from controllers.power_utils import get_avg_power_multiple, get_avg_power, get_weighted_avg_power, get_power_estimates
from controllers.controller_utils import execute_adaptive_model, get_budget_index, ModelResults


CONTROLLER_PATH = 'model-controller-{0}.pkl.gz'
MIN_INIT = 0.7
MAX_INIT = 1.0
REG_NOISE = 0.001
STEAL_ITERATIONS = 5
RANDOM_MOVE = 0.05


ThresholdData = namedtuple('ThresholdData', ['stop_probs', 'model_correct'])


# HELPER FUNCTIONS

def levels_to_execute(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Finds the number of levels to execute for each batch sample using the given stop probabilities
    and stop thresholds.

    Args:
        probs: A [B, L] or [1, B, L] array of stop probabilities for each batch sample (B) and level (L)
        thresholds: A [S, L] array of thresholds for each level (L) and budget (S)
    Returns:
        A [S, B] array containing the level to halt inference at for each sample (B) under each budget (S).
    """
    # Reshape to [1, B, L] array if necessary
    if len(probs.shape) == 2:
        probs = np.expand_dims(probs, axis=0)

    # Validate shapes
    assert probs.shape[2] == thresholds.shape[1], 'Probs ({0}) and thresholds ({1}) must have the same number of levels'.format(probs.shape[2], thresholds.shape[1])

    # Compute the predictions based on this threshold setting. The level predictions are a 0/1
    # array which is 0 when we should NOT use this level and 1 when we should
    expanded_thresholds = np.expand_dims(thresholds, axis=1)  # [S, 1, L]
    level_predictions = (probs > expanded_thresholds).astype(int)  # [S, B, L]

    # Based on these level predictions, we compute the number of levels for each batch sample
    level_idx = np.arange(start=0, stop=thresholds.shape[-1])
    mask = (1.0 - level_predictions) * BIG_NUMBER  # Big number when incorrect, 0 when correct
    index_mask = mask + level_idx  # [S, B, L]
    levels = np.min(index_mask, axis=-1)  # [S, B]
    levels = np.minimum(levels, thresholds.shape[-1] - 1).astype(int)  # Clip the output, [S, B]

    return levels


def classification_for_levels(model_correct: np.ndarray, levels: np.ndarray, batch_idx: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns the correct/not correct classification when executing the levels specified in the given array.

    Args:
        model_correct: A [B, L] array denoting whether the model is right or wrong for each sample (B) and level (L)
        levels: A [S, B] array containing the number of levels to execute for each sample (B) and budget (S)
        batch_idx: An optional [B] array containing the batch indices. This argument is provided purely for efficiency.
    Returns:
        A [S, B] array of predictions for each sample (B) and under each budget (B)
    """
    # Validate shapes
    assert model_correct.shape[0] == levels.shape[1], 'Batch sizes must be aligned between model_correct ({0}) and levels ({1}) arrays'.format(model_correct.shape[0], levels.shape[1])

    # Get useful dimensions
    batch_size = model_correct.shape[0]
    num_budgets = levels.shape[0]

    if batch_idx is None:
        batch_idx = np.arange(start=0, stop=batch_size)

    correct_per_sample: List[np.ndarray] = []
    for i in range(num_budgets):
        level_pred = np.squeeze(model_correct[batch_idx, levels[i, :]])  # [B]
        correct_per_sample.append(level_pred)

    return np.vstack(correct_per_sample)  # [S, B]


def get_level_counts(levels: np.ndarray, num_levels: int) -> np.ndarray:
    """
    Returns the count distribution of levels for each budget.

    Args:
        levels: A [S, B] array of levels for each budget (S) and batch sample (B)
        num_levels: The number of levels (L)
    Returns:
        A [S, L] array of counts for each budget. Element [i, j] of the output array
        denotes the number of samples which end at level j under budget i.
    """
    # Bincount only works for 1-dimensional arrays, so we must perform the bincount for each
    # budget individually.
    num_budgets = levels.shape[0]
    return np.vstack([np.bincount(levels[b], minlength=num_levels) for b in range(num_budgets)])  # [S, L]


def get_budget_interpolation_values(target: float, budgets: np.ndarray, avg_level_counts: np.ndarray, num_levels: int, seq_length: int) -> Tuple[int, int, float]:
    budget_idx = index_of(budgets, target)
    if budget_idx >= 0:
        return budget_idx, budget_idx, 1

    power_multiplier = int(seq_length / num_levels)

    min_power = get_avg_power(num_samples=1, seq_length=seq_length, multiplier=power_multiplier)
    max_power = get_avg_power(num_samples=seq_length, seq_length=seq_length)

    # Get the nearest two budgets nearest two known budgets
    lower_idx, upper_idx = -1, -1
    for idx in range(0, len(budgets) - 1):
        if budgets[idx] <= target and budgets[idx + 1] >= target:
            lower_idx = idx
            upper_idx = idx + 1

    # Compute the expected power for each threshold
    power_estimates = get_power_estimates(num_levels=num_levels, seq_length=seq_length)
    
    expected_power: List[float] = []
    for counts in avg_level_counts:
        expected = np.sum(counts * power_estimates)
        expected_power.append(expected)

    # If the budget is out of the range of the learned budgets, the we supplement the learned
    # thresholds with fixed policies at either end.
    if lower_idx == -1 or upper_idx == -1:
        if target < budgets[0]:

            # The budget is below the lowest learned budget. If it is below the lowest power amount,
            # then we use a fixed policy on the lowest level. Otherwise, we interpolate as usual.
            if target < min_power:
                return -1, -1, 1

            lower_power = min_power
            upper_power = expected_power[0]
        else:
            # The budget is above the highest learned budget. We either fix the policy to the highest level
            # or interpolate given the position of the budget w.r.t. the highest power level.
            if target > max_power:
                return len(budgets), len(budgets), 1

            lower_power = expected_power[-1]
            upper_power = max_power
    else:
        lower_budget = expected_power[lower_idx]
        upper_budget = expected_power[upper_idx]

    if abs(upper_power - lower_power) < SMALL_NUMBER:
        return lower_idx, lower_idx, lower_power

    # Interpolation weight, Clipped to the range [0, 1]
    z = (target - lower_power) / (upper_power - lower_power)
    z = min(max(z, 0), 1)

    return lower_idx, upper_idx, z


# BUDGET OPTIMIZER

class BudgetOptimizer:

    def __init__(self,
                 num_levels: int,
                 budgets: np.ndarray,
                 seq_length: int,
                 precision: int,
                 trials: int,
                 max_iter: int,
                 patience: int,
                 train_frac: float):
        assert train_frac > 0.0 and train_frac <= 1.0, 'Training Fraction must be in (0, 1]'

        self._num_levels = num_levels
        self._num_budgets = budgets.shape[0]
        self._budgets = budgets
        self._precision = precision
        self._trials = trials
        self._max_iter = max_iter
        self._patience = patience
        self._seq_length = seq_length
        self._power_multiplier = int(self._seq_length / self._num_levels)
        self._train_frac = train_frac

        self._thresholds = None

    @property
    def thresholds(self) -> np.ndarray:
        return self._thresholds

    def loss_function(self, thresholds: np.ndarray, budgets: np.ndarray, model_correct: np.ndarray, stop_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the loss of a given set of thresholds on the model results.

        Args:
            thresholds: A [S, L] array of thresholds for each budget (S) and model level (L)
            budgets: A [S] array of budgets
            model_correct: A [B, L] array of binary model correct labels for each batch sample (B) and model level (L)
            stop_probs: A [B, L] array of stop probabilities for each batch sample (B) and model level (L)
        Returns:
            A tuple of two elements:
                (1) A [S] array of scores for each budget
                (2) A [S] array of avg power readings for each budget
        """
        # Compute the number of levels to execute
        levels = levels_to_execute(probs=stop_probs, thresholds=thresholds)  # [S, B]

        # Compute the approximate power
        avg_power = np.vstack([get_avg_power_multiple(levels[idx] + 1, self._seq_length, self._power_multiplier) for idx in range(budgets.shape[0])])  # [S, 1]
        avg_power = np.squeeze(avg_power, axis=-1)  # [S]

        # Compute the accuracy
        correct_per_level = classification_for_levels(model_correct=model_correct, levels=levels)
        accuracy = np.average(correct_per_level, axis=-1)  # [S]

        # Adjust the accuracy by accounting for the budget
        max_time = stop_probs.shape[0]
        time_steps = np.minimum(((budgets * max_time) / avg_power).astype(int), max_time)  # [S]
        adjusted_accuracy = (accuracy * time_steps) / max_time  # [S]

        loss = -adjusted_accuracy

        return loss, avg_power

    def fit(self, train_data: ThresholdData, valid_data: ThresholdData, should_print: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits thresholds to the budgets corresponding to this class.

        Args:
            train_data: A tuple of the following two elements. The given data is used for threshold fitting.
                (1) stop_probs: A [B, L] array of stop probabilities for each level (L) and sample (B)
                (2) model_correct: A [B, L] array of binary labels denoting whether the model level is right (1) or wrong (0)
                                   for each sample.
            valid_data: A tuple of the same structure as train_data. The data is used to detect convergence.
            should_print: Whether we should print the results
        Returns:
            A tuple of three elements.
                (1) A [S] array containing thresholds for each budget (S budgets in total)
                (2) A [S, L] array of the normalized level counts for each budget
                (3) A [S] array containing the final generalization accuracy for each budget
        """
        # Arrays to keep track of the best thresholds per budget
        best_thresholds: List[np.ndarray] = []  # Holds list of best thresholds for each budget
        best_loss: List[float] = []

        power_estimates = get_power_estimates(self._num_levels, self._seq_length)
        level_idx = np.arange(self._num_levels)

        for budget in self._budgets:
            if should_print:
                print('===== Starting Budget {0:.3f} ====='.format(budget))

            # Initialize the random state. We use the same seed for all budgets to get consistent results
            rand = np.random.RandomState(seed=42)

            # Initialize the thresholds uniformly at random. We cap the thresholds at the smallest known level
            # which consumes power more than the budget. This prevents quick, greedy methods to get under the budget
            # in early iterations.
            init_thresholds = rand.uniform(low=MIN_INIT, high=MAX_INIT, size=(self._trials, self._num_levels))
            init_thresholds = round_to_precision(init_thresholds, self._precision)

            power_diff = power_estimates - budget
            diff_mask = (power_diff <= 0).astype(int) * BIG_NUMBER
            max_level = np.argmin(power_diff + diff_mask)

            mask = np.expand_dims((level_idx < max_level).astype(int), axis=0)  # [1, L]
            init_thresholds = init_thresholds * mask

            # Fit thresholds for this budget
            thresholds, loss = self.fit_single(train_data=train_data,
                                               valid_data=valid_data,
                                               init_thresholds=init_thresholds,
                                               budget=budget,
                                               stealing_iterations=STEAL_ITERATIONS,
                                               rand=rand,
                                               should_print=should_print)

            best_thresholds.append(thresholds)
            best_loss.append(loss)

        final_thresholds = np.vstack(best_thresholds)

        # Get the level distribution
        levels = levels_to_execute(probs=valid_data.stop_probs, thresholds=final_thresholds)
        level_counts = get_level_counts(levels=levels, num_levels=self._num_levels)  # [S, L]
        avg_level_counts = level_counts / (np.sum(level_counts, axis=-1, keepdims=True) + SMALL_NUMBER)

        self._thresholds = final_thresholds
        return final_thresholds, avg_level_counts, -np.array(best_loss).reshape(-1)

    def evaluate(self, model_correct: np.ndarray, stop_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the already-fitted thresholds on the given data points.

        Args:
            model_correct: A [B, L] array of binary labels denoting whether the model is correct
                at each batch sample (B) and level (L)
            stop_probs: A [B, L] array of stop probabilities at each sample (B) and level (L)
        Returns:
            A tuple of two elements:
                (1) A [S] array of system accuracy for each budget (S)
                (2) A [S] array of power for each budget (S)
        """
        assert self._thresholds is not None, 'Must fit the optimizer first'
        assert model_correct.shape == stop_probs.shape, 'model_correct ({0}) and stop_probs ({1}) must have the same shape'.format(model_correct.shape, stop_probs.shape)

        # The loss is the negative accuracy AFTER accounting for the budget
        loss, pwr = self.loss_function(model_correct=model_correct,
                                       budgets=self._budgets,
                                       stop_probs=stop_probs,
                                       thresholds=self._thresholds)
        return -loss, pwr

    def fit_single(self,
                   train_data: ThresholdData,
                   valid_data: ThresholdData,
                   init_thresholds: np.ndarray,
                   budget: float,
                   stealing_iterations: int,
                   rand: np.random.RandomState,
                   should_print: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits the optimizer to the given predictions of the logistic regression model and neural network model.

        Args:
            train_data: A pair of arrays representing the inputs and outputs for threshold training. This pair has the following
                structure:
                    (1) model_correct: A [B, L] array of results for each sample (B) and level (L) in the neural network. The results
                        are 0/1 values indicating if this sample was classified correctly (1) or incorrectly (0)
                    (2) stop_probs: A [B, L] array of stop probabilities for each sample (B) and level (L)
            valid_data: A pair of arrays containing the validation set. Same structure as train_data.
            init_thresholds: An [S, L] of initial thresholds for each budget (S) and level (L)
            should_print: Whether we should print intermediate results
        Returns:
            A tuple of two elements:
            (1) A [S, L] array of thresholds for each budget
            (2) A [S, 1] array of loss values for the returned thresholds
        """
        # Copy the initial thresholds, [S, L] array. We set the last threshold to zero because
        # there is no decision to make once inference reaches the top level.
        thresholds = np.copy(init_thresholds)
        thresholds[:, -1] = 0

        # Variables to track convergence and best overall result
        best_thresholds = np.copy(thresholds)  # [S, L]
        early_stopping_counter = np.zeros(thresholds.shape[0])  # [S]
        has_converged = np.zeros(thresholds.shape[0])  # [S]
        best_valid_loss = np.ones(thresholds.shape[0])  # [S]
        best_valid_power = np.zeros(thresholds.shape[0])

        # The number 1 in fixed point representation with the specific precision
        fp_one = 1 << self._precision

        prev_level = None
        budget_array = np.full(fill_value=budget, shape=thresholds.shape[0])

        for i in range(self._max_iter):

            # Select a random level to optimize. We skip the top-level because it does
            # not correspond to a trainable threshold.
            level = rand.randint(low=0, high=self._num_levels - 1)

            # Prevent optimizing the same thresholds twice in a row. This is wasted work.
            if prev_level is not None and level == prev_level:
                level = level - 1 if level > 0 else self._num_levels - 2

            # [S] array of threshold values for the select values
            best_t = np.copy(thresholds[:, level])  # The 'best' are the current thresholds at this level

            # Initialize best loss and best power for this iteration
            best_loss, best_power = self.loss_function(thresholds=thresholds,
                                                       budgets=budget_array,
                                                       model_correct=train_data.model_correct,
                                                       stop_probs=train_data.stop_probs)

            # Select best threshold over the (discrete) search space
            for t in range(fp_one):

                # Set the candidate thresholds
                candidate_value = t / fp_one
                thresholds[:, level] = candidate_value

                # Compute the fitness, both return values are [S] arrays
                loss, avg_power = self.loss_function(thresholds=thresholds,
                                                     budgets=budget_array,
                                                     model_correct=train_data.model_correct,
                                                     stop_probs=train_data.stop_probs)

                has_improved = np.logical_and(loss < best_loss, np.logical_not(has_converged))

                best_t = np.where(has_improved, candidate_value, best_t)
                best_power = np.where(has_improved, avg_power, best_power)
                best_loss = np.where(has_improved, loss, best_loss)

            thresholds[:, level] = best_t  # Set the best thresholds based on the training data

            # Compute the validation loss. The result is a [S] array.
            valid_loss, valid_power = self.loss_function(thresholds=thresholds,
                                                         budgets=budget_array,
                                                         model_correct=valid_data.model_correct,
                                                         stop_probs=valid_data.stop_probs)

            # Detect improvement on the validation set
            has_improved = np.logical_and(valid_loss < best_valid_loss, np.logical_not(has_converged))

            best_thresholds = np.where(np.expand_dims(has_improved, axis=-1), thresholds, best_thresholds)  # [S, L]
            best_valid_loss = np.where(has_improved, valid_loss, best_valid_loss)  # [S]
            best_valid_power = np.where(has_improved, valid_power, best_valid_power)  # [S]

            # Increment early stopping counter
            early_stopping_counter = np.where(has_improved, 0, np.minimum(early_stopping_counter + 1, self._patience))

            # Allow budgets to steal thresholds with improved validation loss. We add a random move
            # to explore the `promising` region
            lowest_loss = np.min(best_valid_loss)
            lowest_idx = np.argmin(best_valid_loss)

            if (i + 1) % stealing_iterations == 0:
                for j in range(best_thresholds.shape[0]):

                    is_improved = lowest_loss < best_valid_loss[j]
                    accuracy_diff = lowest_loss - best_valid_loss[j]


                    if is_improved and lowest_idx != j:
                        random_move = rand.uniform(low=-RANDOM_MOVE, high=RANDOM_MOVE, size=best_thresholds.shape[1])
                        thresholds[j] = round_to_precision(best_thresholds[lowest_idx] + random_move, self._precision)
                        thresholds[j] = np.clip(thresholds[j], a_min=0, a_max=1)

                        early_stopping_counter[j] = 0

            # Detect convergence
            has_converged = (early_stopping_counter >= self._patience)

            prev_level = level

            if should_print:
                print('Completed Iteration: {0}: level {1}'.format(i, level))
                print('\tBest Train Loss: {0}'.format(best_loss))
                print('\tApprox Train Power: {0}'.format(best_power))
                print('\tBest Valid Loss: {0}'.format(best_valid_loss))

            # Terminate early if all budgets have converged
            if has_converged.all():
                if should_print:
                    print('Converged.')
                break

        # We zero out the thresholds at all points after the first zero. This does not
        # change the correctness but prevents mistakes during interpolation.
        zero_comparison = np.isclose(best_thresholds, 0)  # [S, L]

        level_idx = np.expand_dims(np.arange(best_thresholds.shape[1]), axis=0)  # [1, L]
        masked_indices = (1.0 - zero_comparison) * BIG_NUMBER + level_idx  # [S, L]
        mask_index = np.where(np.any(zero_comparison, axis=-1), np.argmin(masked_indices, axis=-1), best_thresholds.shape[1])  # [S]

        threshold_mask = (level_idx < np.expand_dims(mask_index, axis=1)).astype(float)  # [S, L]
        best_thresholds = best_thresholds * threshold_mask

        # Get the best set of thresholds according to the thresholds with the lowest loss.
        # We break ties by selecting the thresholds with the higher power amount.
        final_loss, final_power = self.loss_function(thresholds=best_thresholds,
                                                     budgets=budget_array,
                                                     model_correct=valid_data.model_correct,
                                                     stop_probs=valid_data.stop_probs)

        best_loss = np.min(final_loss)
        equal_mask = np.logical_and(np.isclose(final_loss, best_loss), final_power <= budget)

        if equal_mask.any():
            best_idx = np.argmax(final_power * equal_mask)
        else:
            best_idx = np.argmin(best_loss)

        return best_thresholds[best_idx], final_loss[best_idx]


# Model Controllers

class Controller:

    def fit(self, series: DataSeries, should_print: bool):
        pass

    def predict_sample(self, stop_probs: np.ndarray, budget: int) -> Tuple[int, Optional[float]]:
        raise NotImplementedError()


class AdaptiveController(Controller):

    def __init__(self, model_path: str,
                 dataset_folder: str,
                 precision: int,
                 budgets: List[float],
                 trials: int,
                 patience: int,
                 max_iter: int,
                 train_frac: float):
        self._model_path = model_path
        self._dataset_folder = dataset_folder

        # Load the model and dataset
        model, dataset = restore_neural_network(model_path, dataset_folder=dataset_folder)
        self._model = model
        self._dataset = dataset

        self._num_levels = model.num_outputs
        self._seq_length = model.metadata[SEQ_LENGTH]
        self._num_classes = model.metadata[NUM_CLASSES]
        self._budgets = np.array(list(sorted(budgets)))
        self._num_budgets = len(self._budgets)
        self._precision = precision
        self._trials = trials
        self._patience = patience
        self._max_iter = max_iter
        self._train_frac = train_frac

        self._thresholds = None
        self._est_accuracy = None
        self._validation_accuracy = None
        self._fit_start_time: Optional[str] = None
        self._fit_end_time: Optional[str] = None
        self._is_fitted = False

        # Create the budget optimizer
        self._budget_optimizer = BudgetOptimizer(num_levels=self._num_levels,
                                                 seq_length=self._seq_length,
                                                 budgets=self._budgets,
                                                 precision=self._precision,
                                                 trials=self._trials,
                                                 patience=patience,
                                                 max_iter=max_iter,
                                                 train_frac=train_frac)

    @property
    def thresholds(self) -> np.ndarray:
        return self._thresholds

    @property
    def budgets(self) -> np.ndarray:
        return self._budgets

    def load_validation_accuracy(self, validation_accuracy: np.ndarray):
        self._validation_accuracy = validation_accuracy  # [L]

    def get_estimated_accuracy(self, budget: float) -> float:
        """
        Returns the estimated accuracy for the given budget based on the results of threshold fitting.
        """
        budget_idx = index_of(self.budgets, budget)

        assert budget_idx >= 0, 'Unknown budget: {0}'.format(budget)
        assert self._est_accuracy is not None, 'Must call fit() first'

        return self._est_accuracy[budget_idx]

    def fit(self, series: DataSeries, should_print: bool):
        start_time = datetime.now()

        # Execute the model on the training, validation, and testing sets
        train_results = execute_adaptive_model(self._model, self._dataset, series=DataSeries.TEST)
        valid_results = execute_adaptive_model(self._model, self._dataset, series=DataSeries.VALID)
        test_results = execute_adaptive_model(self._model, self._dataset, series=DataSeries.TEST)

        train_correct = train_results.predictions == train_results.labels  # [N, L]
        valid_correct = valid_results.predictions == valid_results.labels  # [K, L]
        test_correct = test_results.predictions == test_results.labels  # [M, L]

        # We optimize the thresholds on the validation set, and then use a subset of the training
        # set to detect convergence. We perform optimization in this manner because the validation
        # set is 'unseen' from the model's point of view. Splitting the validation into subsets, however,
        # can leads to a data shortage during optimization. Thus, we compromise by using the original
        # training set to as validation for the thresholds.
        rand = np.random.RandomState(seed=341)
        sample_idx = np.arange(train_correct.shape[0])
        rand.shuffle(sample_idx)

        train_idx = sample_idx[:valid_correct.shape[0]]

        # Create the training and validation sets for the optimization. See above comment
        # for explanation behind switching the roles of the original training and validation sets.
        train_data = ThresholdData(model_correct=valid_correct,
                                   stop_probs=valid_results.stop_probs)
        valid_data = ThresholdData(model_correct=train_correct[train_idx, :],
                                   stop_probs=train_results.stop_probs[train_idx, :])

        # Fit the thresholds
        self._thresholds, self._avg_level_counts, _ = self._budget_optimizer.fit(train_data=train_data,
                                                                                 valid_data=valid_data,
                                                                                 should_print=should_print)
        end_time = datetime.now()

        # Evaluate the model optimizer
        train_acc, _ = self._budget_optimizer.evaluate(model_correct=train_data.model_correct,
                                                       stop_probs=train_data.stop_probs)
        valid_acc, valid_pwr = self._budget_optimizer.evaluate(model_correct=valid_data.model_correct,
                                                               stop_probs=valid_data.stop_probs)
        test_acc, _ = self._budget_optimizer.evaluate(model_correct=test_correct,
                                                      stop_probs=test_results.stop_probs)

        print('Train Accuracy: {0}'.format(train_acc))
        print('Valid Accuracy: {0}'.format(valid_acc))
        print('Test Accuracy: {0}'.format(test_acc))

        print('Approximate Power: {0}'.format(valid_pwr))

        self._fit_start_time = start_time.strftime('%Y-%m-%d-%H-%M-%S')
        self._fit_end_time = end_time.strftime('%Y-%m-%d-%H-%M-%S')

        self._est_accuracy = valid_acc  # Set the estimated accuracy to the validation accuracy

        self._stop_means = np.average(valid_data.stop_probs, axis=0)  # [L]
        self._stop_std = np.std(valid_data.stop_probs, axis=0)  # [L]


        # Estimate the level distribution for each predicted label.
        valid_predictions = train_results.predictions[train_idx, :]
        levels = levels_to_execute(valid_data.stop_probs, self._thresholds)  # [S, K]
        pred = classification_for_levels(valid_predictions, levels)  # [S, K]

        self._label_distribution: Dict[float, Dict[int, np.ndarray]] = dict()
        for budget_idx, budget in enumerate(self._budgets):
            budget_dict: Dict[int, np.ndarray] = dict()

            budget_levels = levels[budget_idx]
            budget_predictions = pred[budget_idx]

            for class_idx in range(self._num_classes):
                class_mask = np.isclose(budget_predictions, class_idx).astype(int)
                class_level_counts = np.bincount(budget_levels, minlength=self._num_levels, weights=class_mask)  # [L]
                budget_dict[class_idx] = class_level_counts

            self._label_distribution[budget] = budget_dict

        # Add estimates for the lowest and highest fixed policies
        lowest_thresholds = np.zeros(self._num_levels)
        highest_thresholds = np.ones(self._num_levels)

        levels = levels_to_execute(valid_data.stop_probs, np.vstack([lowest_thresholds, highest_thresholds]))  # [2, K]
        pred = classification_for_levels(valid_predictions, levels)  # [2, K]

        self._lowest_distribution: Dict[int, np.ndarray] = dict()
        for class_idx in range(self._num_classes):
            class_mask = np.isclose(pred[0], class_idx).astype(int)
            class_level_counts = np.bincount(levels[0], minlength=self._num_levels, weights=class_mask)
            self._lowest_distribution[class_idx] = class_level_counts

        self._highest_distribution: Dict[int, np.ndarray] = dict()
        for class_idx in range(self._num_classes):
            class_mask = np.isclose(pred[1], class_idx).astype(int)
            class_level_counts = np.bincount(levels[1], minlength=self._num_levels, weights=class_mask)
            self._highest_distribution[class_idx] = class_level_counts

        self._is_fitted = True

    def estimate_level_distribution(self, budget: float) -> Dict[int, np.ndarray]:
        assert self._label_distribution is not None, 'Must call fit() first'

        lower_idx, upper_idx, weight = get_budget_interpolation_values(target=budget,
                                                                       budgets=self._budgets,
                                                                       avg_level_counts=self._avg_level_counts,
                                                                       num_levels=self._num_levels,
                                                                       seq_length=self._seq_length)
        if lower_idx < 0:
            lower_counts = self._lowest_distribution
        elif lower_idx >= self._num_levels:
            lower_counts = self._highest_distribution
        else:
            lower_counts = self._label_distribution[self._budgets[lower_idx]]

        if upper_idx < 0:
            upper_counts = self._lowest_distribution
        elif upper_idx >= self._num_levels:
            upper_counts = self._highest_distribution
        else:
            lower_counts = self._label_distribution[self._budgets[upper_idx]]

        result: Dict[int, np.ndarray] = dict()
        for label in range(self._num_classes):
            result[label] = lower_counts[label] * (1 - weight) + upper_counts[label] * weight

        return result

    def get_thresholds(self, budget: float) -> np.ndarray:
        assert self._thresholds is not None, 'Must call fit() first'

        power_multiplier = int(self._seq_length / self._num_levels)

        # If the budget is above the power needed for the highest validation accuracy
        # we cap the execution to these thresholds. This makes the execution more amenable
        # to the controller, as the controller will force power as close as possible to the
        # budget. This doesn't work when the model has out-of-order accuracy.

        best_idx = np.argmax(self._est_accuracy)
        if self._budgets[best_idx] <= budget:
            return self._thresholds[best_idx]

        # Check if this budget is known based from the optimization phase
        budget_idx = index_of(self._budgets, value=budget)

        min_power = get_avg_power(num_samples=1, seq_length=self._seq_length, multiplier=power_multiplier)
        max_power = get_avg_power(num_samples=self._seq_length, seq_length=self._seq_length)

        # If we already have the budget, then use the corresponding thresholds
        if budget_idx >= 0:
            return self._thresholds[budget_idx]

        # Otherwise, we interpolate the thresholds from the nearest two known budgets
        #lower_budget_idx, upper_budget_idx = None, None
        #for idx in range(0, len(self._budgets) - 1):
        #    if self._budgets[idx] < budget and self._budgets[idx + 1] > budget:
        #        lower_budget_idx = idx
        #        upper_budget_idx = idx + 1

        ## Compute the expected power for each threshold
        #power_estimates = get_power_estimates(num_levels=self._num_levels,
        #                                      seq_length=self._seq_length)
        #expected_power: List[float] = []
        #for counts in self._avg_level_counts:
        #    expected = np.sum(counts * power_estimates)
        #    expected_power.append(expected)

        ## If the budget is out of the range of the learned budgets, the we supplement the learned
        ## thresholds with fixed policies at either end.
        #if lower_budget_idx is None or upper_budget_idx is None:
        #    if budget < self._budgets[0]:
        #        # The budget is below the lowest learned budget. If it is below the lowest power amount,
        #        # then we use a fixed policy on the lowest level. Otherwise, we interpolate as usual.
        #        fixed_thresholds = np.zeros_like(self._thresholds[0])
        #        if budget < min_power:
        #            return fixed_thresholds

        #        lower_budget = min_power
        #        # upper_budget = self._budgets[0]
        #        upper_budget = expected_power[0]

        #        lower_thresh = fixed_thresholds
        #        upper_thresh = self._thresholds[0]
        #    else:
        #        # The budget is above the highest learned budget. We either fix the policy to the highest level
        #        # or interpolate given the position of the budget w.r.t. the highest power level.
        #        fixed_thresholds = np.ones_like(self._thresholds[0])
        #        fixed_thresholds[-1] = 0
        #        if budget > max_power:
        #            return fixed_thresholds

        #        lower_budget = expected_power[-1]
        #        #lower_budget = self._budgets[-1]
        #        upper_budget = max_power

        #        lower_thresh = self._thresholds[-1]
        #        upper_thresh = fixed_thresholds
        #else:
        #    # lower_budget = self._budgets[lower_budget_idx]
        #    # upper_budget = self._budgets[upper_budget_idx]

        #    lower_budget = expected_power[lower_budget_idx]
        #    upper_budget = expected_power[upper_budget_idx]

        #    lower_thresh = self._thresholds[lower_budget_idx]
        #    upper_thresh = self._thresholds[upper_budget_idx]

        #if abs(upper_budget - lower_budget) < SMALL_NUMBER:
        #    return lower_thresh

        ## Interpolation weight, Clipped to the range [0, 1]
        #z = (budget - lower_budget) / (upper_budget - lower_budget)
        #z = min(max(z, 0), 1)

        lower_idx, upper_idx, weight = get_budget_interpolation_values(budgets=self._budgets,
                                                                       target=budget,
                                                                       avg_level_counts=self._avg_level_counts,
                                                                       num_levels=self._num_levels,
                                                                       seq_length=self._seq_length)
        if lower_idx < 0:
            lower_thresh = np.zeros(self._thresholds.shape[1])
        elif lower_idx > self._thresholds.shape[1]:
            lower_thresh = np.ones(self._thresholds.shape[1])
        else:
            lower_thresh = self._thresholds[lower_idx]
        
        if upper_idx < 0:
            upper_thresh = np.zeros(self._thresholds.shape[1])
        elif upper_idx > self._thresholds.shape[1]:
            upper_thresh = np.ones(self._thresholds.shape[1])
        else:
            upper_thresh = self._thresholds[upper_idx]
        
        # Create thresholds and projected budget
        thresholds = lower_thresh * (1 - weight) + upper_thresh * weight
        thresholds[-1] = 0

        # Round to fixed point representation
        thresholds = round_to_precision(thresholds, precision=self._precision)

        return thresholds

    def get_avg_level_counts(self, budget: int) -> np.ndarray:
        budget_idx = index_of(self._budgets, value=budget)
        assert budget_idx >= 0, 'Could not find values for budget {0}'.format(budget)

        return self._avg_level_counts[budget_idx]

    def predict_sample(self, stop_probs: np.ndarray, budget: int) -> Tuple[int, Optional[float]]:
        """
        Predicts the number of levels given the list of hidden states. The states are assumed to be in order.

        Args:
            stop_probs: An array of [L] stop probabilities, one for each level
            budget: The budget to perform inference under. This controls the employed thresholds.
        Returns:
            A tuple of two elements:
                (1) The number of levels to execute
                (2) The power consumed while executing this number of levels
        """
        assert self._is_fitted, 'Model is not fitted'

        # Infer the thresholds for this budget
        thresholds = self.get_thresholds(budget)

        power_mult = int(self._seq_length / self._num_levels)

        for level, stop_prob in enumerate(stop_probs):
            if thresholds[level] < stop_prob:
                power = get_avg_power(level + 1, self._seq_length, power_mult)
                return level, power

        # By default, we return the top level
        return self._num_levels - 1, get_avg_power(self._seq_length, self._seq_length)

    def evaluate(self, budget: float, model_results: ModelResults) -> Tuple[float, float]:
        assert self._thresholds is not None, 'Must call fit() first'

        max_time = model_results.stop_probs.shape[0]

        thresholds = self.get_thresholds(budget=budget)
        thresholds = np.expand_dims(thresholds, axis=0)  # [1, L]

        levels = levels_to_execute(probs=model_results.stop_probs,
                                   thresholds=thresholds)  # [1, N]
        predictions = classification_for_levels(model_correct=model_results.predictions,
                                                levels=levels)  # [1, N]
        predictions = predictions.reshape(-1, 1).astype(int)  # [N, 1]
        correct = (predictions == model_results.labels).astype(float)  # [N]
        accuracy = np.average(correct)

        avg_power = get_avg_power_multiple(num_samples=np.squeeze(levels + 1, axis=0),
                                           seq_length=self._seq_length,
                                           multiplier=int(self._seq_length / self._num_levels))
        time_steps = min(int((budget * max_time) / avg_power), max_time)
        adjusted_accuracy = (accuracy * time_steps) / max_time

        return float(adjusted_accuracy), float(avg_power)

    def as_dict(self):
        return {
            'budgets': self._budgets,
            'thresholds': self._thresholds,
            'trials': self._trials,
            'is_fitted': self._is_fitted,
            'model_path': self._model_path,
            'dataset_folder': self._dataset_folder,
            'precision': self._precision,
            'patience': self._patience,
            'max_iter': self._max_iter,
            'train_frac': self._train_frac,
            'avg_level_counts': self._avg_level_counts,
            'est_accuracy': self._est_accuracy,
            'stop_means': self._stop_means,
            'stop_std': self._stop_std,
            'label_distribution': self._label_distribution,
            'lowest_distribution': self._lowest_distribution,
            'highest_distribution': self._highest_distribution,
            'fit_start_time': self._fit_start_time,
            'fit_end_time': self._fit_end_time
        }

    def save(self, output_file: Optional[str] = None):
        """
        Serializes the model into a pickle file.
        """
        # Create a default file name if none is given
        if output_file is None:
            save_folder, model_path = os.path.split(self._model_path)
            model_name = extract_model_name(model_path)
            output_file = os.path.join(save_folder, CONTROLLER_PATH.format(model_name))

        # Save the model components
        save_pickle_gz(self.as_dict(), output_file)

    @classmethod
    def load(cls, save_file: str, dataset_folder: Optional[str] = None, model_path: Optional[str] = None):
        """
        Loads the controller from the given serialized file.
        """
        # Load the serialized information.
        serialized_info = read_pickle_gz(save_file)
        dataset_folder = dataset_folder if dataset_folder is not None else serialized_info['dataset_folder']
        model_path = model_path if model_path is not None else serialized_info['model_path']

        # Initialize the new controller
        controller = AdaptiveController(model_path=model_path,
                                        dataset_folder=dataset_folder,
                                        precision=serialized_info['precision'],
                                        budgets=serialized_info['budgets'],
                                        trials=serialized_info['trials'],
                                        patience=serialized_info.get('patience', 10),
                                        max_iter=serialized_info.get('max_iter', 100),
                                        train_frac=serialized_info.get('train_frac', 0.7))

        # Set remaining fields
        controller._thresholds = serialized_info['thresholds']
        controller._avg_level_counts = serialized_info['avg_level_counts']
        controller._is_fitted = serialized_info['is_fitted']
        controller._fit_start_time = serialized_info.get('fit_start_time')
        controller._fit_end_time = serialized_info.get('fit_end_time')
        controller._est_accuracy = serialized_info.get('est_accuracy')
        controller._stop_means = serialized_info.get('stop_means')
        controller._stop_std = serialized_info.get('stop_std')
        controller._label_distribution = serialized_info.get('label_distribution')
        controller._lowest_distribution = serialized_info.get('lowest_distribution')
        controller._highest_distribution = serialized_info.get('highest_distribution')

        return controller


class FixedController(Controller):

    def __init__(self, model_index: int):
        self._model_index = model_index

    def predict_sample(self, stop_probs: np.ndarray, budget: int) -> Tuple[int, Optional[float]]:
        """
        Predicts the label for the given inputs. This strategy always uses the same index.
        """
        return self._model_index, None


class RandomController(Controller):

    def __init__(self, budgets: List[float], seq_length: int, num_levels: int):
        self._budgets = np.array(budgets)
        self._seq_length = seq_length
        self._num_levels = num_levels
        self._power_multiplier = int(seq_length / num_levels)
        self._levels = np.arange(num_levels)

        self._threshold_dict: Dict[float, np.ndarray] = dict()

        # Create random state for reproducible results
        self._rand = np.random.RandomState(seed=62)
        self._is_fitted = False

    def fit(self, series: DataSeries, should_print: bool = False):
        # Fit a weighted average for each budget
        thresholds: List[np.ndarray] = []
        power_array = np.array([get_avg_power(i+1, self._seq_length, self._power_multiplier) for i in range(self._num_levels)])

        for budget in self._budgets:
            distribution = PowerDistribution(power_array, target=budget)
            self._threshold_dict[budget] = distribution.fit()

        self._is_fitted = True

    def predict_sample(self, stop_probs: np.ndarray, budget: int) -> Tuple[int, Optional[float]]:
        """
        Predicts the number of levels given the list of hidden states. The states are assumed to be in order.

        Args:
            stop_probs: An [L] array of stop probabilities
            budget: The budget to perform inference under. This controls the employed thresholds.
        Returns:
            A tuple of two elements:
                (1) The number of levels to execute
                (2) The power consumed when executing this number of levels
        """
        assert self._is_fitted, 'Model is not fitted'

        # Get thresholds for this budget if needed to infer
        thresholds = self._threshold_dict[budget]
        chosen_level = self._rand.choice(self._levels, p=thresholds)
        return chosen_level, get_avg_power(chosen_level + 1, seq_length=self._seq_length, multiplier=self._power_multiplier)


class MultiModelController(Controller):

    def __init__(self, sample_counts: List[np.ndarray], model_accuracy: List[float], seq_length: int, max_time: int, allow_violations: bool):
        model_power: List[float] = []
        for counts in sample_counts:
            power = get_weighted_avg_power(counts, seq_length=seq_length)
            model_power.append(power)

        self._model_power = np.array(model_power).reshape(-1)
        self._model_accuracy = np.array(model_accuracy).reshape(-1)
        self._max_time = max_time
        self._allow_violations = allow_violations

    def predict_sample(self, stop_probs: np.ndarray, budget: float) -> Tuple[int, Optional[float]]:
        """
        Predicts the number of levels given the list of hidden states. The states are assumed to be in order.

        Args:
            stop_probs: An [L] array of stop probabilities
            budget: The budget to perform inference under. This controls the employed thresholds.
        Returns:
            A tuple of two elements:
                (1) The number of levels to execute
                (2) The power consumed when executing this number of levels
        """
        model_idx = get_budget_index(budget=budget,
                                     valid_accuracy=self._model_accuracy,
                                     max_time=self._max_time,
                                     power_estimates=self._model_power,
                                     allow_violations=self._allow_violations)

        return model_idx, self._model_power[model_idx]


class BudgetWrapper:

    def __init__(self, model_predictions: np.ndarray, controller: Controller, max_time: int, seq_length: int, num_classes: int, num_levels: int, budget: float):
        self._controller = controller
        self._num_classes = num_classes
        self._seq_length = seq_length
        self._num_levels = num_levels
        self._model_predictions = model_predictions
        self._power_budget = budget
        self._energy_budget = budget * max_time

        # Save variables corresponding to the budget
        self._max_time = max_time
        self._energy_sum = 0.0
        self._power: List[float] = []
        self._energy_margin = 0.05  # Small margin to prevent going over the budget unknowingly

    def predict_sample(self, stop_probs: np.ndarray, current_time: int, budget: int, noise: float) -> Tuple[Optional[int], int, float]:
        """
        Predicts the label for the given inputs.

        Args:
            stop_probs: An [L] array of the stop probabilities for each level.
            current_time: The current time index
            budget: The power budget
            noise: The noise on the power reading
        Returns:
            A tuple of three elements:
                (1) A classification for the t-th sample (given by current time). This may be None when the system
                    has exhausted the energy budget
                (2) The number of levels used during execution
                (3) The average power consumed to produce this classification
        """
        # Calculate used energy to determine whether to use the model
        used_energy = self.get_consumed_energy()
        should_use_controller = bool(used_energy < self._energy_budget - self._energy_margin)

        # By acting randomly, we incur no energy (no need to collect input samples)
        if not should_use_controller:
            pred = None
            level = 0
            power = 0.0
        else:
            # If not acting randomly, we use the neural network to perform the classification.
            level, model_power = self._controller.predict_sample(stop_probs=stop_probs, budget=budget)

            # If no power is given, we default to using the known power estimates.
            if model_power is None:
                power = get_avg_power(level + 1, seq_length=self._seq_length, multiplier=int(self._seq_length / self._num_levels))
            else:
                power = model_power

            power = power + noise
            pred = self._model_predictions[current_time, level]

        # If this inference pushes the system over budget, we actually cannot complete it. We thus clip
        # the energy and clear the prediction.
        if self._energy_sum > self._energy_budget:
            pred = None
            power = self._energy_budget - self._energy_sum

        # Log the energy consumption
        self._power.append(power)
        self._energy_sum += power

        return pred, level, power

    @property
    def power(self) -> List[float]:
        return self._power

    def get_consumed_energy(self) -> float:
        return self._energy_sum

    def reset(self):
        self._power = []
        self._energy_sum = 0.0


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-paths', type=str, nargs='+', required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--budgets', type=float, nargs='+', required=True)
    parser.add_argument('--precision', type=int, required=True)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--train-frac', type=float, default=0.7)
    parser.add_argument('--should-print', action='store_true')
    args = parser.parse_args()

    for model_path in args.model_paths:
        print('Starting model at {0}'.format(model_path))

        # Create the adaptive model
        controller = AdaptiveController(model_path=model_path,
                                        dataset_folder=args.dataset_folder,
                                        precision=args.precision,
                                        budgets=args.budgets,
                                        trials=args.trials,
                                        patience=args.patience,
                                        max_iter=args.max_iter,
                                        train_frac=args.train_frac)

        # Fit the model on the validation set
        controller.fit(series=DataSeries.VALID, should_print=args.should_print)
        controller.save()
