import numpy as np
import os.path
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from typing import List, Optional, Tuple, Dict

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from utils.np_utils import index_of, round_to_precision
from utils.constants import OUTPUT, BIG_NUMBER, SMALL_NUMBER, INPUTS, SEQ_LENGTH, DROPOUT_KEEP_RATE, SEQ_LENGTH
from utils.file_utils import save_pickle_gz, read_pickle_gz, extract_model_name
from utils.loading_utils import restore_neural_network
from controllers.power_distribution import PowerDistribution
from controllers.power_utils import get_avg_power_multiple, get_avg_power, get_weighted_avg_power
from controllers.controller_utils import execute_adaptive_model, get_budget_index, ModelResults


CONTROLLER_PATH = 'model-controller-{0}.pkl.gz'
MARGIN = 1000
MIN_INIT = 0.7
MAX_INIT = 1.0


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
        assert train_frac > 0.0 and train_frac < 1.0, 'Training Fraction must be in (0, 1)'

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
        self._valid_frac = 1.0 - train_frac

        self._rand = np.random.RandomState(seed=42)
        self._thresholds = None

    @property
    def thresholds(self) -> np.ndarray:
        return self._thresholds

    def loss_function(self, thresholds: np.ndarray, model_correct: np.ndarray, stop_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the loss of a given set of thresholds on the model results.

        Args:
            thresholds: A [S, L] array of thresholds for each budget (S) and model level (L)
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
        avg_power = np.vstack([get_avg_power_multiple(levels[idx] + 1, self._seq_length, self._power_multiplier) for idx in range(self._num_budgets)])  # [S, 1]
        avg_power = np.squeeze(avg_power, axis=-1)  # [S]

        # Compute the accuracy
        correct_per_level = classification_for_levels(model_correct=model_correct, levels=levels)
        accuracy = np.average(correct_per_level, axis=-1)  # [S]

        # Adjust the accuracy by accounting for the budget
        max_time = stop_probs.shape[0]
        time_steps = np.minimum(((self._budgets * max_time) / avg_power).astype(int), max_time)  # [S]
        adjusted_accuracy = (accuracy * time_steps) / max_time  # [S]

        loss = -adjusted_accuracy

        return loss, avg_power

    def fit(self, stop_probs: np.ndarray, model_correct: np.ndarray, should_print: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits thresholds to the budgets corresponding to this class.

        Args:
            stop_probs: A [B, L] array of stop probabilities for each level (L) and sample (B)
            model_correct: A [B, L] array of binary labels denoting whether the model level is right (1) or wrong (0) for each
                sample.
            should_print: Whether we should print the results
        Returns:
            A tuple of two elements.
                (1) A [S] array containing thresholds for each budget (S budgets in total)
                (2) A [S, L] array of the normalized level counts for each budget
        """
        # Validate shapes of input arrays
        assert stop_probs.shape[1] == self._num_levels, 'Stop Probs array has wrong number of levels ({0}). Expected {1}'.format(stop_probs.shape[1], self._num_levels)
        assert model_correct.shape[1] == self._num_levels, 'Model Correct array has wrong number of levels ({0}). Expected {1}'.format(model_correct.shape[1], self._num_levels)
        assert stop_probs.shape[0] == model_correct.shape[0], 'Stop Probs ({0}) and Model Correct ({1}) must have same first dimension'.format(stop_probs.shape[0], model_correct.shape[0])

        # Arrays to keep track of the best thresholds per budget
        best_thresholds = np.ones(shape=(self._num_budgets, self._num_levels))  # [S, L]
        best_loss = np.ones(shape=(self._num_budgets, 1), dtype=float)  # [S, 1]

        # Randomly split data into training and validation folds
        num_samples = stop_probs.shape[0]
        split_point = int(self._train_frac * num_samples)
        
        sample_idx = np.arange(num_samples)
        self._rand.shuffle(sample_idx)
        train_idx, valid_idx = sample_idx[:split_point], sample_idx[split_point:]
        
        train_data = ThresholdData(model_correct=model_correct[train_idx, :],
                                   stop_probs=stop_probs[train_idx, :])
        valid_data = ThresholdData(model_correct=model_correct[valid_idx, :],
                                   stop_probs=stop_probs[valid_idx, :])

        for t in range(self._trials):
            if should_print:
                print('===== Starting Trial {0} ====='.format(t))

            # Initialize the thresholds uniformly at random. We sort the thresholds in decreasing
            # order to align with the notion of answer 'confidence'. This is a heuristic.
            init_thresholds = self._rand.uniform(low=MIN_INIT, high=MAX_INIT, size=(self._num_budgets, self._num_levels))
            init_thresholds = round_to_precision(init_thresholds, self._precision)
            init_thresholds = np.flip(np.sort(init_thresholds, axis=-1), axis=-1)  # [S, L]

            # Fit the thresholds ([S, L]) and get the corresponding loss ([S, 1])
            thresholds, loss = self.fit_single(train_data=train_data,
                                               valid_data=valid_data,
                                               init_thresholds=init_thresholds,
                                               should_print=should_print)

            # Set the thresholds using the best seen fitness so far
            best_thresholds = np.where(loss < best_loss, thresholds, best_thresholds)
            best_loss = np.where(loss < best_loss, loss, best_loss)

            if should_print:
                print('Completed Trial {0}. Best Loss: {1}'.format(t, best_loss))

        # Get the level distribution
        levels = levels_to_execute(probs=stop_probs, thresholds=best_thresholds)
        level_counts = get_level_counts(levels=levels, num_levels=self._num_levels)  # [S, L]
        avg_level_counts = level_counts / (np.sum(level_counts, axis=-1, keepdims=True) + SMALL_NUMBER)

        self._thresholds = best_thresholds
        return best_thresholds, avg_level_counts

    def evaluate(self, model_correct: np.ndarray, stop_probs: np.ndarray) -> np.ndarray:
        """
        Evaluates the already-fitted thresholds on the given data points.

        Args:
            model_correct: A [B, L] array of binary labels denoting whether the model is correct
                at each batch sample (B) and level (L)
            stop_probs: A [B, L] array of stop probabilities at each sample (B) and level (L)
        Returns:
            A [S] array of system accuracy for each budget (S)
        """
        assert self._thresholds is not None, 'Must fit the optimizer first'
        assert model_correct.shape == stop_probs.shape, 'model_correct ({0}) and stop_probs ({1}) must have the same shape'.format(model_correct.shape, stop_probs.shape)

        # The loss is the negative accuracy AFTER accounting for the budget
        loss, _ = self.loss_function(model_correct=model_correct,
                                     stop_probs=stop_probs,
                                     thresholds=self._thresholds)
        return -loss

    def fit_single(self, train_data: ThresholdData, valid_data: ThresholdData, init_thresholds: np.ndarray, should_print: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits the optimizer to the given predictions of the logistic regression model and neural network model.

        Args:
            train_data: A pair of arrays representing the inputs and outputs for threshold training. This pair has the following
                structure:
                    (1) model_correct: A [B, L] array of results for each sample (B) and level (L) in the neural network. The results
                        are 0/1 values indicating if this sample was classified correctly (1) or incorrectly (0)
                    (2) stop_probs: A [B, L] array of stop probabilities for each sample (B) and level (L)
            valid_data: A pair of arrays holding the inputs and outputs for threshold validation. Uses the same structure
                as train_data
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

        # The number 1 in fixed point representation with the specific precision
        fp_one = 1 << self._precision

        # Variables to detect convergence based on validation results
        best_thresholds = np.copy(thresholds)  # [S, L]
        best_validation_loss = np.ones((self._num_budgets, ))  # [S]
        has_converged = np.zeros((self._num_budgets, ))  # [S]
        early_stopping_counter = np.zeros((self._num_budgets, ))  # [S]

        for i in range(self._max_iter):

            # Select a random level to optimize. We skip the top-level because it does
            # not correspond to a trainable threshold.
            level = self._rand.randint(low=0, high=self._num_levels - 1)

            # [S] array of threshold values for the select values
            best_t = np.copy(thresholds[:, level])  # The 'best' are the current thresholds at this level
            best_loss = np.ones(shape=(self._num_budgets, ), dtype=float)  # [S]
            best_power = np.zeros_like(best_loss)

            # Create the start values to enable a interval of size [MARGIN] within [0, 1]
            fp_init = (best_t * fp_one).astype(int)  # Convert to fixed point
            end_values = np.minimum(fp_init + int((MARGIN + 1) / 2), fp_one)  # Clip to one
            start_values = np.maximum(end_values - MARGIN, 0)

            # Select best threshold over the (discrete) search space
            for offset in range(MARGIN):
                
                # Compute the predictions using the threshold on the logistic regression model
                candidate_values = np.minimum((start_values + offset) / fp_one, 1)  # [S]
                thresholds[:, level] = candidate_values

                # Compute the fitness, both return values are [S] arrays
                loss, avg_power = self.loss_function(thresholds=thresholds,
                                                     model_correct=train_data.model_correct,
                                                     stop_probs=train_data.stop_probs)
                
                has_improved = np.logical_and(loss < best_loss, np.logical_not(has_converged))

                best_t = np.where(has_improved, candidate_values, best_t)
                best_power = np.where(has_improved, avg_power, best_power)
                best_loss = np.where(has_improved, loss, best_loss)

            thresholds[:, level] = best_t  # Set the best thresholds based on the training data

            # Evaluate the thresholds on the validation set, return values are [S] arrays
            validation_loss, _ = self.loss_function(thresholds=thresholds,
                                                    model_correct=valid_data.model_correct,
                                                    stop_probs=valid_data.stop_probs)

            # Set validation thresholds and best loss based on improvement on the validation set
            loss_comparison = np.logical_and(validation_loss < best_validation_loss, np.logical_not(has_converged))  # [S]

            best_thresholds = np.where(np.expand_dims(loss_comparison, axis=1), thresholds, best_thresholds)  # [S, L]
            best_validation_loss = np.where(loss_comparison, validation_loss, best_validation_loss)  # [S]

            early_stopping_counter = np.where(loss_comparison, 0, np.minimum(early_stopping_counter + 1, self._patience))
            has_converged = np.logical_or(early_stopping_counter >= self._patience, has_converged).astype(int)

            if should_print:
                print('Completed Iteration: {0}: level {1}'.format(i, level))
                print('\tBest Train Loss: {0}'.format(best_loss))
                print('\tApprox Train Power: {0}'.format(best_power))
                print('\tBest Valid Loss: {0}'.format(best_validation_loss))

            # Terminate early if all budgets have converged
            if has_converged.all():
                if should_print:
                    print('Converged.')
                break

        final_loss, _ = self.loss_function(thresholds=best_thresholds,
                                           model_correct=valid_data.model_correct,
                                           stop_probs=valid_data.stop_probs)

        return best_thresholds, np.expand_dims(final_loss, axis=-1)


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
        self._budgets = np.array(list(sorted(budgets)))
        self._num_budgets = len(self._budgets)
        self._precision = precision
        self._trials = trials
        self._patience = patience
        self._max_iter = max_iter
        self._train_frac = train_frac

        self._thresholds = None
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

    def fit(self, series: DataSeries, should_print: bool):
        start_time = datetime.now()

        train_results = execute_adaptive_model(self._model, self._dataset, series=series)
        test_results = execute_adaptive_model(self._model, self._dataset, series=DataSeries.TEST)

        train_correct = train_results.predictions == train_results.labels  # [N, L]
        test_correct = test_results.predictions == test_results.labels  # [M, L]

        # Fit the thresholds
        self._thresholds, self._avg_level_counts = self._budget_optimizer.fit(model_correct=train_correct,
                                                                              stop_probs=train_results.stop_probs,
                                                                              should_print=should_print)

        
        end_time = datetime.now()
        
        # Evaluate the model optimizer
        train_acc = self._budget_optimizer.evaluate(model_correct=train_correct, stop_probs=train_results.stop_probs)
        test_acc = self._budget_optimizer.evaluate(model_correct=test_correct, stop_probs=test_results.stop_probs)

        print('Train Accuracy: {0}'.format(train_acc))
        print('Test Accuracy: {0}'.format(test_acc))

        self._fit_start_time = start_time.strftime('%Y-%m-%d-%H-%M-%S')
        self._fit_end_time = end_time.strftime('%Y-%m-%d-%H-%M-%S')

        self._is_fitted = True

    def get_thresholds(self, budget: int) -> np.ndarray:
        assert self._thresholds is not None, 'Must call fit() first'

        budget_idx = index_of(self._budgets, value=budget)

        min_power = get_avg_power(num_samples=1, seq_length=self._seq_length, multiplier=int(self._seq_length / self._num_levels))
        max_power = get_avg_power(num_samples=self._seq_length, seq_length=self._seq_length)

        # If we already have the budget, then use the corresponding thresholds
        if budget_idx >= 0:
            return self._thresholds[budget_idx]

        # Otherwise, we interpolate the thresholds from the nearest two known budgets
        lower_budget_idx, upper_budget_idx = None, None
        for idx in range(0, len(self._budgets) - 1):
            if self._budgets[idx] < budget and self._budgets[idx + 1] > budget:
                lower_budget_idx = idx
                upper_budget_idx = idx + 1

        # If the budget is out of the range of the learned budgets, the we supplement the learned
        # thresholds with fixed policies at either end.
        if lower_budget_idx is None or upper_budget_idx is None:
            if budget < self._budgets[0]:
                # The budget is below the lowest learned budget. If it is below the lowest power amount,
                # then we use a fixed policy on the lowest level. Otherwise, we interpolate as usual.
                fixed_thresholds = np.zeros_like(self._thresholds[0])
                if budget < min_power:
                    return fixed_thresholds

                lower_budget = min_power
                upper_budget = self._budgets[0]

                lower_thresh = fixed_thresholds
                upper_thresh = self._thresholds[0]
            else:
                # The budget is above the highest learned budget. We either fix the policy to the highest level
                # or interpolate given the position of the budget w.r.t. the highest power level.
                fixed_thresholds = np.ones_like(self._thresholds[0])
                fixed_thresholds[-1] = 0
                if budget > max_power:
                    return fixed_thresholds

                lower_budget = self._budgets[-1]
                upper_budget = max_power

                lower_thresh = self._thresholds[-1]
                upper_thresh = fixed_thresholds
        else:
            lower_budget = self._budgets[lower_budget_idx]
            upper_budget = self._budgets[upper_budget_idx]

            lower_thresh = self._thresholds[lower_budget_idx]
            upper_thresh = self._thresholds[upper_budget_idx]

        # Interpolation weight
        z = (budget - lower_budget) / (upper_budget - lower_budget)

        # Create thresholds
        thresholds = lower_thresh * (1 - z) + upper_thresh * z

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

        thresholds = np.expand_dims(self.get_thresholds(budget=budget), axis=0)  # [1, L]
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
