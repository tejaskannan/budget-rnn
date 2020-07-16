import numpy as np
import os.path
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from threshold_optimization.optimize_thresholds import get_serialized_info
from utils.rnn_utils import get_logits_name, get_states_name, AdaptiveModelType, get_input_name, is_cascade, is_sample
from utils.np_utils import index_of, round_to_precision
from utils.constants import OUTPUT, BIG_NUMBER, SMALL_NUMBER, INPUTS, SEQ_LENGTH, DROPOUT_KEEP_RATE
from utils.file_utils import save_pickle_gz, read_pickle_gz, extract_model_name
from controllers.distribution_prior import DistributionPrior


POWER = np.array([24.085, 32.776, 37.897, 43.952, 48.833, 50.489, 54.710, 57.692, 59.212, 59.251])
VIOLATION_FACTOR = 0.01
UNDERSHOOT_FACTOR = 0.01
CONTROLLER_PATH = 'model-logistic-controller-{0}.pkl.gz'
MARGIN = 1000
MIN_INIT = 0.8
MAX_INIT = 1.0
C = 0.01
NOISE = 0.01


def get_power_for_levels(power: np.ndarray, num_levels: int) -> np.ndarray:
    assert num_levels <= len(power), 'Must have fewer levels than power estimates'    

    if len(power) == num_levels:
        return power

    median_index = int(len(power) / 2)
    start_index = median_index - int(num_levels / 2)
    end_index = start_index + num_levels
    return power[start_index:end_index]


def fetch_model_states(model: AdaptiveModel, dataset: Dataset, series: DataSeries):
    logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]
    state_ops = [get_states_name(i) for i in range(model.num_outputs)]
    stop_output_ops = ['stop_output_{0}'.format(i) for i in range(model.num_outputs)]

    data_generator = dataset.minibatch_generator(series=series,
                                                 batch_size=model.hypers.batch_size,
                                                 metadata=model.metadata,
                                                 should_shuffle=False)
    # Lists to keep track of model results
    labels: List[np.ndarray] = []
    states: List[np.ndarray] = []
    stop_outputs: List[np.ndarray] = []
    level_predictions: List[np.ndarray] = []
    level_logits: List[np.ndarray] = []

    # Index of state to use for stop/start prediction
    states_index = 0
    if is_cascade(model.model_type):
        states_index = -1

    seq_length = model.metadata[SEQ_LENGTH]
    num_sequences = model.num_sequences

    for batch_num, batch in enumerate(data_generator):
        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False, epoch_num=0)
        model_results = model.execute(feed_dict, logit_ops + state_ops + stop_output_ops)

        first_states = np.concatenate([np.expand_dims(np.squeeze(model_results[op][states_index]), axis=1) for op in state_ops], axis=1)  # [B, D]

        inputs = np.array(batch[INPUTS])
        states.append(first_states)

        # Concatenate logits into a [B, L, C] array (logit_ops is already ordered by level).
        # For reference, L is the number of levels and C is the number of classes
        logits_concat = np.concatenate([np.expand_dims(model_results[op], axis=1) for op in logit_ops], axis=1)
        level_logits.append(logits_concat)

        # Compute the predictions for each level
        level_pred = np.argmax(logits_concat, axis=-1)  # [B, L]
        level_predictions.append(level_pred)

        true_values = np.squeeze(batch[OUTPUT])
        labels.append(true_values)

        batch_stop_outputs = np.concatenate([np.expand_dims(model_results[op], axis=1) for op in stop_output_ops], axis=1)  # [B, T]
        stop_outputs.append(batch_stop_outputs)

    states = np.concatenate(states, axis=0)
    level_predictions = np.concatenate(level_predictions, axis=0)
    labels = np.concatenate(labels, axis=0).reshape(-1, 1)
    level_logits = np.concatenate(level_logits, axis=0)
    stop_outputs = np.concatenate(stop_outputs, axis=0)

    y = (level_predictions == labels).astype(float)
    print('Level Accuracy: {0}'.format(np.average(y, axis=0)))

    return states, y, level_logits, labels, stop_outputs


def levels_to_execute(logistic_probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    # Compute the predictions based on this threshold setting. The level predictions are a 0/1
    # array which is 0 when we should NOT use this level and 1 when we should
    expanded_thresholds = np.expand_dims(thresholds, axis=1)  # [S, 1, L]
    level_predictions = (logistic_probs > expanded_thresholds).astype(int)  # [S, B, L]

    # Based on these level predictions, we compute the number of levels for each batch sample
    level_idx = np.arange(start=0, stop=thresholds.shape[-1])
    mask = (1.0 - level_predictions) * BIG_NUMBER  # Big number when incorrect, 0 when correct
    index_mask = mask + level_idx  # [S, B, L]
    levels = np.min(index_mask, axis=-1)  # [S, B]
    levels = np.minimum(levels, thresholds.shape[-1] - 1).astype(int)  # Clip the output, [S, B]

    return levels


def adjust_thresholds(clf_predictions: np.ndarray, thresholds: np.ndarray, target_distribution: np.ndarray, precision: int) -> np.ndarray:
    fp_one = 1 << precision
    thresholds = np.copy(thresholds)
    num_levels = thresholds.shape[1]
    num_budgets = thresholds.shape[0]

    clf_predictions = np.expand_dims(clf_predictions, axis=0)  # [1, B, L]

    for level in range(num_levels):
        levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=thresholds)
        level_counts = np.vstack([np.bincount(levels[i, :], minlength=num_levels) for i in range(num_budgets)])  # [S, L]
        level_fractions = level_counts / np.sum(level_counts, axis=-1, keepdims=True)  # [S, L]

        direction = 1 - 2 * (target_distribution[:, level] > level_fractions[:, level]).astype(float)

        i = 0
        while (direction * (target_distribution[:, level] - level_fractions[:, level]) <= 0).all() and i < fp_one:
            thresholds[:, level] += (direction / fp_one)
            thresholds = np.clip(thresholds, a_min=0, a_max=1)

            levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=thresholds)

            # Compute the approximate power and accuracy
            level_counts = np.vstack([np.bincount(levels[i, :], minlength=num_levels) for i in range(num_budgets)])  # [S, L]
            level_fractions = level_counts / np.sum(level_counts, axis=-1, keepdims=True)  # [S, L]
            i += 1

    thresholds[:, level] -= (direction / fp_one)
    final_levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=thresholds)
    level_counts = np.vstack([np.bincount(final_levels[i, :], minlength=num_levels) for i in range(num_budgets)])  # [S, L]
    final_distribution = level_counts / np.sum(level_counts, axis=-1, keepdims=True)  # [S, L]

    return thresholds


def level_errors(logistic_probs: np.ndarray, thresholds: np.ndarray, network_predictions: np.ndarray) -> np.ndarray:
    """
    Calculates the distribution of threshold errors for each model level. This is used purely for debugging.

    Args:
        logistic_probs: A [1, B, L] array of logistic regression probabilities.
        thresholds: A [S, L] array of learned thresholds
        network_predictions: A [1, B, L] array of 0/1 classifications for each level.
    """
    expanded_thresholds = np.expand_dims(thresholds, axis=1)  # [S, 1, L]
    level_diff = logistic_probs - expanded_thresholds  # [S, B, L]

    levels = levels_to_execute(logistic_probs, thresholds)  # [S, B]

    for budget_idx, budget_thresholds in enumerate(thresholds):  # [S]
        print(budget_thresholds)
        for level in range(thresholds.shape[1] - 1):  # [L]
            is_incorrect = (1.0 - network_predictions[:, level])  # [B]
            is_correct = network_predictions[:, level]  # [B]
            prob_diff = level_diff[budget_idx, :, level]  # [B]
            chosen_levels = (levels[budget_idx, :] == level).astype(float)  # [B]

            logistic_variance = np.square(np.std(logistic_probs[:, level]))
            logistic_avg = np.average(logistic_probs[:, level])

            incorrect_mask = is_incorrect * chosen_levels
            incorrect_diff = incorrect_mask * prob_diff
            avg_inc_diff = np.sum(incorrect_diff) / np.maximum(np.sum(incorrect_mask), SMALL_NUMBER)

            correct_mask = is_correct * chosen_levels
            correct_diff = correct_mask * prob_diff
            avg_cor_diff = np.sum(correct_diff) / np.maximum(np.sum(correct_mask), SMALL_NUMBER)

            print('Average Gap on Level {0}: Incorrect -> {1:.5f}, Correct -> {2:.5f}, Prob Avg (Var): {3:.5f} ({4:.5f})'.format(level, avg_inc_diff, avg_cor_diff, logistic_avg, logistic_variance))


def predictions_for_levels(model_predictions: np.ndarray, levels: np.ndarray, batch_idx: np.ndarray) -> np.ndarray:
    preds_per_sample: List[np.ndarray] = []
    for i in range(levels.shape[0]):
        level_pred = np.squeeze(model_predictions[batch_idx, levels[i, :]])
        preds_per_sample.append(level_pred)

    preds_per_sample = np.vstack(preds_per_sample)  # [S, B]
    return preds_per_sample


def fit_anneal_rate(start_value: float, end_value: float, steps: int):
    return np.exp((1.0 / steps) * np.log(max(end_value, SMALL_NUMBER) / max(start_value, SMALL_NUMBER)))


### Budget optimizer classes ###

class BudgetOptimizer:
    
    def __init__(self, num_levels: int, budgets: np.ndarray, precision: int, trials: int, max_iter: int, min_iter: int, patience: int, power: np.ndarray):
        self._num_levels = num_levels
        self._num_budgets = budgets.shape[0]
        self._budgets = budgets
        self._precision = precision
        self._trials = trials
        self._max_iter = max_iter
        self._patience = patience
        self._rand = np.random.RandomState(seed=42)
        self._thresholds = None
        self._min_iter = min_iter
        self._power = power

    def fit(self, network_predictions: np.ndarray, clf_predictions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def evaluate(self, network_predictions: np.ndarray, clf_predictions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_approx_power(self, levels: np.ndarray) -> np.ndarray:
        """
        Approximates the power consumption given profiled power results.

        Args:
            levels: A [S, B] array of the levels for each sample (B) and budget (S)
        Returns:
            An [S] array containing the average power consumption for each budget thresholds.
        """
        level_counts = np.vstack([np.bincount(levels[i, :], minlength=self._num_levels) for i in range(self._num_budgets)])  # [S, L]
        normalized_level_counts = level_counts / np.sum(level_counts, axis=-1, keepdims=True)  # [S, L]
        approx_power = np.sum(normalized_level_counts * self._power, axis=-1).astype(float)  # [S]
        return approx_power, normalized_level_counts


class SimulatedAnnealingOptimizer(BudgetOptimizer):

    def __init__(self, num_levels: int, budgets: np.ndarray, precision: int, trials: int, max_iter: int, patience: int, temp: float, anneal_rate: float):
        super().__init__(num_levels, budgets, precision, trials, max_iter, patience)
        self._max_iter = max_iter
        self._temp = temp
        self._anneal_rate = anneal_rate

    def fit(self, network_results: np.ndarray, clf_predictions: np.ndarray):
        # Expand the clf predictions for later broadcasting
        clf_predictions = np.expand_dims(clf_predictions, axis=0)  # [1, B, L]

        # Initialize thresholds, [S, L] array
        thresholds = round_to_precision(self._rand.uniform(low=0.2, high=0.8, size=(self._num_budgets, self._num_levels)), self._precision)
        thresholds = np.flip(np.sort(thresholds, axis=-1), axis=-1)
        thresholds[:, -1] = 0

        # The number 1 in fixed point representation
        fp_one = 1 << self._precision

        # Array of level indices
        level_idx = np.arange(start=0, stop=self._num_levels).reshape(1, 1, -1)  # [1, 1, L]
        batch_idx = np.arange(start=0, stop=clf_predictions.shape[1])  # [B]

        # Variable for convergence
        early_stopping_counter = 0

        best_fitness = np.zeros(shape=(self._num_budgets,), dtype=float)
        best_power = np.zeros_like(best_fitness)
        margin = 0.4
        temp = self._temp

        for i in range(self._max_iter):
            prev_thresholds = np.copy(thresholds)
    
            # Generate a random move
            random_move = round_to_precision(self._rand.uniform(low=-margin, high=margin, size=thresholds.shape), self._precision)
            random_move[:, -1] = 0

            candidate_thresholds = np.clip(thresholds + random_move, a_min=0, a_max=1)

            levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=candidate_thresholds)

            # Compute the approximate power and accuracy
            approx_power, _ = self.get_approx_power(levels=levels)
            dual_term = approx_power - self._budgets  # [S]
            dual_penalty = np.where(dual_term > 0, VIOLATION_FACTOR * dual_term, -UNDERSHOOT_FACTOR * dual_term)

            correct_per_level = predictions_for_levels(model_predictions=network_results, levels=levels, batch_idx=batch_idx)
            accuracy = np.average(correct_per_level, axis=-1)  # [S]

            # Compute the fitness (we aim to maximize this objective)
            fitness = accuracy - dual_penalty

            # Determine when to set thresholds based on fitness and temperature
            random_move_prob = self._rand.uniform(low=0.0, high=1.0, size=(self._num_budgets, ))
            fitness_diff = fitness - best_fitness  # [S]
            temperature_prob = np.exp(-1 * fitness_diff / temp)

            selection = np.logical_or(fitness_diff > 0, temperature_prob > random_move_prob)
            best_fitness = np.where(selection, fitness, best_fitness)
            best_power = np.where(selection, approx_power, best_power)
            thresholds = np.where(selection, candidate_thresholds, thresholds)

            # Anneal the temperature
            temp = temp * self._anneal_rate

            print('Completed iteration {0}: Fitness -> {1}'.format(i+1, best_fitness))

            if np.isclose(thresholds, prev_thresholds).all():
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0

            if early_stopping_counter >= self._patience:
                print('Converged.')
                break

        return thresholds


class CoordinateOptimizer(BudgetOptimizer):

    def fitness_function(self, thresholds: np.ndarray, network_results: np.ndarray, clf_predictions: np.ndarray, batch_size: int, violation_factor: float, undershoot_factor: float):
        # Compute the number of levels to execute
        levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=thresholds)  # [B]

        # Compute the approximate power
        approx_power, normalized_level_counts = self.get_approx_power(levels=levels)
        dual_term = approx_power - self._budgets  # [S]
        dual_penalty = np.where(dual_term > 0, violation_factor, undershoot_factor) * np.square(dual_term)

        # Compute the accuracy
        batch_idx = np.arange(start=0, stop=batch_size)  # [B]
        correct_per_level = predictions_for_levels(model_predictions=network_results, levels=levels, batch_idx=batch_idx)

        accuracy = np.average(correct_per_level, axis=-1)  # [S]

        return -accuracy + dual_penalty, approx_power

    def fit(self, network_results: np.ndarray, clf_predictions: np.ndarray):
        best_thresholds = np.ones(shape=(self._num_budgets, self._num_levels))
        best_fitness = np.ones(shape=(self._num_budgets, 1), dtype=float)

        # Reshape the validation arrays
        valid_clf_predictions = np.expand_dims(clf_predictions, axis=0)  # [1, B, L]

        for t in range(self._trials):
            print('===== Starting Trial {0} ====='.format(t))

            init_thresholds = np.random.uniform(low=MIN_INIT, high=MAX_INIT, size=(self._num_budgets, self._num_levels))
            init_thresholds = round_to_precision(init_thresholds, self._precision)
            init_thresholds = np.flip(np.sort(init_thresholds, axis=-1), axis=-1)  # [S, L]

            thresholds = self.fit_single(network_results=network_results,
                                         clf_predictions=clf_predictions,
                                         init_thresholds=init_thresholds)

            # Compute the fitness
            fitness, _ = self.fitness_function(thresholds=thresholds,
                                               network_results=network_results,
                                               clf_predictions=valid_clf_predictions,
                                               batch_size=valid_clf_predictions.shape[1],
                                               violation_factor=VIOLATION_FACTOR,
                                               undershoot_factor=UNDERSHOOT_FACTOR)
            fitness = np.expand_dims(fitness, axis=1)

            best_thresholds = np.where(fitness < best_fitness, thresholds, best_thresholds)
            best_fitness = np.where(fitness < best_fitness, fitness, best_fitness)
            print('Completed Trial {0}. Best Fitness: {1}'.format(t, best_fitness))

        levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=thresholds)
        level_counts = np.vstack([np.bincount(levels[i, :], minlength=self._num_levels) for i in range(self._num_budgets)])  # [S, L]
        avg_level_counts = level_counts / np.sum(level_counts, axis=-1, keepdims=True)

        self._thresholds = best_thresholds
        return best_thresholds, avg_level_counts

    def print_accuracy_for_levels(self, clf_predictions: np.ndarray, network_results: np.ndarray, thresholds: np.ndarray):
        levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=thresholds)
        level_counts = np.vstack([np.bincount(levels[i, :], minlength=self._num_levels) for i in range(self._num_budgets)])  # [S, L]
        avg_level_counts = level_counts / np.sum(level_counts, axis=-1, keepdims=True)

        batch_idx = np.arange(levels.shape[1])  # [B]
        correct_per_level = predictions_for_levels(model_predictions=network_results, levels=levels, batch_idx=batch_idx)

        print('Level Counts: {0}'.format(avg_level_counts))

        # Calculate the accuracy for each level
        for i in range(self._num_levels):
            level_mask = (levels == i).astype(float)  # [S, B]
            level_correct = np.sum(correct_per_level * level_mask, axis=-1)  # [S]
            level_accuracy = level_correct / (np.sum(level_mask, axis=-1) + SMALL_NUMBER)  # [S]

            print('Accuracy when stopping at level {0}: {1}'.format(i, level_accuracy))

    def evaluate(self, network_results: np.ndarray, clf_predictions: np.ndarray) -> np.ndarray:
        """
        Evaluates the already-fitted thresholds on the given data points.
        """
        assert self._thresholds is not None, 'Must fit the optimizer first'

        # Compute the number of levels to execute per sample
        levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=self._thresholds)

        # Compute the accuracy for each budget
        batch_size = network_results.shape[0]
        batch_idx = np.arange(start=0, stop=batch_size)  # [B]
        correct_per_level = predictions_for_levels(model_predictions=network_results, levels=levels, batch_idx=batch_idx)

        accuracy = np.average(correct_per_level, axis=-1)  # [S]

        return accuracy

    def fit_single(self, network_results: np.ndarray, clf_predictions: np.ndarray, init_thresholds: np.ndarray) -> np.ndarray:
        """
        Fits the optimizer to the given predictions of the logistic regression model and neural network model.

        Args:
            network_results: A [B, L] array of results for each sample and level in the neural network. The results
                are 0/1 values indicating if this sample was classified correctly (1) or incorrectly (0)
            clf_predictions: A [B, L] array of classifications by the logistic regression model.
            patience: Number of trials without change to detect convergence.
        """
        # Expand the clf predictions for later broadcasting
        clf_predictions = np.expand_dims(clf_predictions, axis=0)  # [1, B, L]

        # Copy the initial thresholds, [S, L] array
        thresholds = np.copy(init_thresholds)
        thresholds[:, -1] = 0

        # The number 1 in fixed point representation
        fp_one = 1 << self._precision

        # Variable for convergence
        early_stopping_counter = 0
        prev_thresholds = np.copy(thresholds)

        # best_fitness = np.ones(shape=(self._num_budgets,), dtype=float)
        # best_power = np.zeros_like(best_fitness)

        # Initialize penalty parameters
        violation_factor = 1e-4
        entropy_factor = 1e-4
        undershoot_factor = 1e-4

        violation_anneal_rate = fit_anneal_rate(start_value=violation_factor, end_value=VIOLATION_FACTOR, steps=self._min_iter)
        undershoot_anneal_rate = fit_anneal_rate(start_value=undershoot_factor, end_value=UNDERSHOOT_FACTOR, steps=self._min_iter)

        for i in range(self._max_iter):

            # Select a random level to run
            level = self._rand.randint(low=0, high=self._num_levels - 1)

            # [S] array of threshold values
            best_t = np.copy(thresholds[:, level])  # The 'best' are the previous thresholds at this level
            best_fitness = np.ones(shape=(self._num_budgets,), dtype=float)
            best_power = np.zeros_like(best_fitness)
           
            # Create the start values to enable a interval of size [MARGIN] within [0, 1]
            fp_init = (best_t * fp_one).astype(int)
            end_values = np.minimum(fp_init + int((MARGIN + 1) / 2), fp_one)
            start_values = np.maximum(end_values - MARGIN, 0)

            # start_values = np.maximum((best_t * fp_one).astype(int) - int(MARGIN / 2), 0)

            # Variables for tie-breaking
            steps = np.zeros_like(best_fitness)
            prev_level_fitness = np.ones_like(best_fitness)
            prev_level_approx_power = np.zeros_like(best_power)
            current_thresholds = np.zeros_like(best_t)  # [S]

            # print('Starting threshold: {0}'.format(best_t))

            for offset in range(MARGIN):

                # Compute the predictions using the threshold on the logistic regression model
                candidate_values = np.minimum((start_values + offset) / fp_one, 1)
                thresholds[:, level] = candidate_values

                # Compute the fitness
                fitness, approx_power = self.fitness_function(thresholds=thresholds,
                                                              network_results=network_results,
                                                              clf_predictions=clf_predictions,
                                                              batch_size=clf_predictions.shape[1],
                                                              violation_factor=violation_factor,
                                                              undershoot_factor=undershoot_factor)

                # print('Fitness: {0}, Candidate Value: {1}'.format(fitness, candidate_values))

                # Initialize variables on first iteration
                #if offset == 0:
                #    prev_level_fitness = np.copy(fitness)
                #    prev_level_approx_power = np.copy(approx_power)
                #    current_thresholds = np.copy(thresholds[:, level])

                ## Set the best values at inflection points in the fitness
                #offset_condition = np.full(shape=fitness.shape, fill_value=(offset == MARGIN - 1 or offset == 0))
                #is_fitness_same = np.isclose(prev_level_fitness, fitness)

                #should_set = np.logical_and(prev_level_fitness <= best_fitness, \
                #                            np.logical_or(np.logical_not(is_fitness_same), offset_condition))

                #median_thresholds = np.clip(current_thresholds + ((0.5 * steps).astype(int) / fp_one), a_min=0.0, a_max=1.0)
                #best_t = np.where(should_set, median_thresholds, best_t)  # Set the thresholds to the median amount
                #best_power = np.where(should_set, prev_level_approx_power, best_power)
                #best_fitness = np.where(should_set, prev_level_fitness, best_fitness)

                ## If the fitness is equal to the previous fitness, then we add to the steps.
                ## Otherwise, we reset.
                #steps = np.where(np.isclose(prev_level_fitness, fitness), steps + 1, 0)

                ## Reset variables
                #current_thresholds = np.where(np.logical_not(is_fitness_same), thresholds[:, level], current_thresholds)
                #prev_level_fitness = np.copy(fitness)
                #prev_level_approx_power = np.copy(approx_power)

                best_t = np.where(fitness < best_fitness, candidate_values, best_t)
                best_power = np.where(fitness < best_fitness, approx_power, best_power)
                best_fitness = np.where(fitness < best_fitness, fitness, best_fitness)

            thresholds[:, level] = best_t  # Set the best thresholds
            print('Completed Iteration: {0}: level {1}'.format(i, level))
            print('\tBest Fitness: {0}'.format(-1 * best_fitness))
            print('\tApprox Power: {0}'.format(best_power))
            # print('\tThresholds: {0}'.format(thresholds))

            if i >= self._min_iter and (np.isclose(thresholds, prev_thresholds)).all():
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0

            if early_stopping_counter >= self._patience:
                print('Converged.')
                break

            if i < self._min_iter:
                violation_factor = violation_factor * violation_anneal_rate
                undershoot_factor = undershoot_factor * undershoot_anneal_rate

            prev_thresholds = np.copy(thresholds)

        return thresholds


### Model Controllers ###

class Controller:

    def __init__(self, model_path: str,
                 dataset_folder: str,
                 share_model: bool,
                 precision: int,
                 budgets: List[float],
                 trials: int,
                 power: np.ndarray,
                 budget_optimizer_type: str,
                 patience: int,
                 max_iter: int,
                 min_iter: int):
        self._model_path = model_path
        self._dataset_folder = dataset_folder

        # Load the model and dataset
        model, dataset, _ = get_serialized_info(model_path, dataset_folder=dataset_folder)

        self._model = model
        self._dataset = dataset
        self._is_fitted = False
        self._share_model = share_model
        self._num_levels = model.num_outputs

        self._budgets = np.array(budgets)
        self._num_budgets = len(self._budgets)
        self._precision = precision
        self._trials = trials
        self._thresholds = None
        self._patience = patience
        self._max_iter = max_iter
        self._min_iter = min_iter

        self._power = get_power_for_levels(power, self._num_levels)

        # Create the budget optimizer
        self._budget_optimizer_type = budget_optimizer_type.lower()
        if self._budget_optimizer_type == 'coordinate':
            self._budget_optimizer = CoordinateOptimizer(num_levels=self._num_levels,
                                                         budgets=self._budgets,
                                                         precision=self._precision,
                                                         trials=self._trials,
                                                         patience=patience,
                                                         max_iter=max_iter,
                                                         min_iter=min_iter,
                                                         power=self._power)
        elif self._budget_optimizer_type == 'sim-anneal':
            self._budget_optimizer = SimulatedAnnealingOptimizer(num_levels=self._num_levels,
                                                                 budgets=self._budgets,
                                                                 precision=self._precision,
                                                                 trials=self._trials,
                                                                 temp=0.1,
                                                                 anneal_rate=0.95,
                                                                 patience=patience,
                                                                 max_iter=max_iter,
                                                                 min_iter=min_iter,
                                                                 power=self._power)
        else:
            raise ValueError('Unknown budget optimizer: {0}'.format(budget_optimizer_type))

    def fit(self, series: DataSeries):
        X_train, y_train, train_logits, train_labels, clf_predictions = fetch_model_states(self._model, self._dataset, series=series)
        X_test, y_test, test_logits, test_labels, test_clf_predictions = fetch_model_states(self._model, self._dataset, series=DataSeries.TEST)

        # Fit the thresholds
        self._thresholds, self._avg_level_counts = self._budget_optimizer.fit(network_results=y_train, clf_predictions=clf_predictions)
    
        # Evaluate the model optimizer
        print('======')
        train_acc = self._budget_optimizer.evaluate(network_results=y_train, clf_predictions=clf_predictions)
        test_acc = self._budget_optimizer.evaluate(network_results=y_test, clf_predictions=test_clf_predictions)

        print('Train Accuracy: {0}'.format(train_acc))
        self._budget_optimizer.print_accuracy_for_levels(network_results=y_train, clf_predictions=clf_predictions, thresholds=self._thresholds)

        print('Test Accuracy: {0}'.format(test_acc))
        self._budget_optimizer.print_accuracy_for_levels(network_results=y_test, clf_predictions=test_clf_predictions, thresholds=self._thresholds)

        print('=====')

        self._is_fitted = True

    def get_thresholds(self, budget: int) -> np.ndarray:
        budget_idx = index_of(self._budgets, value=budget)
        assert budget_idx >= 0, 'Could not find values for budget {0}'.format(budget)

        return self._thresholds[budget_idx]

    def get_avg_level_counts(self, budget: int) -> np.ndarray:
        budget_idx = index_of(self._budgets, value=budget)
        assert budget_idx >= 0, 'Could not find values for budget {0}'.format(budget)

        return self._avg_level_counts[budget_idx]

    def predict_sample(self, inputs: np.ndarray, budget: int, thresholds: Optional[np.ndarray] = None) -> int:
        """
        Predicts the number of levels given the list of hidden states. The states are assumed to be in order.

        Args:
            inputs: An array of inputs for this sequence
            budget: The budget to perform inference under. This controls the employed thresholds.
            thresholds: Optional set of thresholds to use. This argument overrides the inferred thresholds.
        Returns:
            The number of levels to execute.
        """
        assert self._is_fitted, 'Model is not fitted'

        # Get thresholds for this budget
        if thresholds is None:
            # Infer the thresholds for this budget
            thresholds = self.get_thresholds(budget)

        stop_output_ops = ['stop_output_{0}'.format(i) for i in range(self._model.num_outputs)]
      
        # Create the input feed dict
        seq_length = self._model.metadata[SEQ_LENGTH]
        num_sequences = self._model.num_sequences
        samples_per_seq = int(seq_length / num_sequences)
        feed_dict = dict()
        for i in range(self._model.num_outputs):
            input_ph = self._model.placeholders[get_input_name(i)]
            if is_sample(self._model.model_type):
                seq_indexes = list(range(i, seq_length, num_sequences))
                sample_tensor = inputs[seq_indexes]
                feed_dict[input_ph] = np.expand_dims(sample_tensor, axis=0)  # Make batch size 1
            else:  # Cascade
                start, end = i * samples_per_seq, (i+1) * samples_per_seq
                sample_tensor = inputs[start:end]
                feed_dict[input_ph] = np.expand_dims(sample_tensor, axis=0)  # Make batch size 1

        # Supply dropout (needed for Adaptive NBOW)
        feed_dict[self._model.placeholders[DROPOUT_KEEP_RATE]] = 1.0

        model_result = self._model.execute(ops=stop_output_ops, feed_dict=feed_dict)
        for level, op_name in enumerate(stop_output_ops):
            stop_prob = model_result[op_name]

            if thresholds[level] < stop_prob:
                return level

        # By default, we return the top level
        return self._num_levels - 1

    def predict_levels(self, series: DataSeries, budget: float) -> Tuple[np.ndarray, np.ndarray]:
        assert self._is_fitted, 'Model is not fitted'

        budget_idx = index_of(self._budgets, value=budget)
        assert budget_idx >= 0, 'Could not find values for budget {0}'.format(budget)

        X, ypred, logits, _, clf_predictions = fetch_model_states(self._model, self._dataset, series=series)
        level_predictions = np.argmax(logits, axis=-1)  # [B, L]

        levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=self._thresholds)

        batch_idx = np.arange(level_predictions.shape[0])
        predictions = predictions_for_levels(model_predictions=level_predictions,
                                             levels=levels,
                                             batch_idx=batch_idx)

        return levels[budget_idx].astype(int), predictions[budget_idx].astype(int)

    def as_dict(self):
        return {
            'budgets': self._budgets,
            'thresholds': self._thresholds,
            'trials': self._trials,
            'is_fitted': self._is_fitted,
            'model_path': self._model_path,
            'dataset_folder': self._dataset_folder,
            'share_model': self._share_model,
            'precision': self._precision,
            'patience': self._patience,
            'max_iter': self._max_iter,
            'min_iter': self._min_iter,
            'budget_optimizer_type': self._budget_optimizer_type,
            'avg_level_counts': self._avg_level_counts
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
    def load(cls, save_file: str):
        """
        Loads the controller from the given serialized file.
        """
        # Load the serialized information.
        serialized_info = read_pickle_gz(save_file)

        # Initialize the new controller
        controller = Controller(model_path=serialized_info['model_path'],
                                dataset_folder=serialized_info['dataset_folder'],
                                share_model=serialized_info['share_model'],
                                precision=serialized_info['precision'],
                                budgets=serialized_info['budgets'],
                                trials=serialized_info['trials'],
                                budget_optimizer_type=serialized_info['budget_optimizer_type'],
                                patience=serialized_info.get('patience', 10),
                                max_iter=serialized_info.get('max_iter', 100),
                                min_iter=serialized_info.get('min_iter', 20),
                                power=POWER)

        # Set remaining fields
        controller._thresholds = serialized_info['thresholds']
        controller._avg_level_counts = serialized_info['avg_level_counts']
        controller._is_fitted = serialized_info['is_fitted']

        return controller


class RandomController(Controller):

    def __init__(self, model_path: str, dataset_folder: str, budgets: List[float], power: np.ndarray):
        self._model_path = model_path
        self._dataset_folder = dataset_folder

        # Load the model and dataset
        self._model, self._dataset, _ = get_serialized_info(model_path, dataset_folder=dataset_folder)

        self._share_model = False
        self._precision = 0
        self._budgets = np.array(budgets)
        self._trials = 0
        self._power = power
        self._budget_optimizer_type = 'random'
        self._patience = 0
        self._max_iter = 0
        self._min_iter = 0

        # Create random state for reproducible results
        self._rand = np.random.RandomState(seed=62)

    def fit(self, series: DataSeries):
        # Fit a weighted average for each budget
        thresholds: List[np.ndarray] = []
        for budget in self._budgets:
            distribution = DistributionPrior(self._power, target=budget)
            distribution.make()
            distribution.init()
            
            thresholds.append(distribution.fit())

        self._thresholds = np.vstack(thresholds)

        self._is_fitted = True

    def predict_sample(self, inputs: np.ndarray, budget: int, thresholds: Optional[np.ndarray] = None) -> int:
        """
        Predicts the number of levels given the list of hidden states. The states are assumed to be in order.

        Args:
            inputs: An array of inputs for this sequence
            budget: The budget to perform inference under. This controls the employed thresholds.
            thresholds: Optional set of thresholds to use. This argument overrides the inferred thresholds.
        Returns:
            The number of levels to execute.
        """
        assert self._is_fitted, 'Model is not fitted'

        # Get thresholds for this budget if needed to infer
        if thresholds is None:
            thresholds = self.get_thresholds(budget)

        levels = np.arange(thresholds.shape[0])  # [L]
        return self._rand.choice(levels, p=thresholds)

    def predict_levels(self, series: DataSeries, budget: float) -> Tuple[np.ndarray, np.ndarray]:
        assert self._is_fitted, 'Model is not fitted'

        budget_idx = index_of(self._budgets, value=budget)
        assert budget_idx >= 0, 'Could not find values for budget {0}'.format(budget)

        _, _, logits, _, _ = fetch_model_states(self._model, self._dataset, series=series)
        level_predictions = np.argmax(logits, axis=-1)  # [B, L]

        budget_distribution = self._thresholds[budget_idx]
        levels_idx = np.arange(budget_distribution.shape[0])  # [L]
        levels = self._rand.choice(levels_idx, p=budget_distribution, size=level_predictions.shape[0])  # [B]

        batch_idx = np.arange(level_predictions.shape[0])
        predictions = predictions_for_levels(model_predictions=level_predictions,
                                             levels=np.expand_dims(levels, axis=0),
                                             batch_idx=batch_idx)

        return levels.astype(int), predictions[budget_idx].astype(int)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-paths', type=str, nargs='+')
    parser.add_argument('--dataset-folder', type=str)
    parser.add_argument('--budgets', type=float, nargs='+')
    parser.add_argument('--precision', type=int, required=True)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--min-iter', type=int, default=20)
    parser.add_argument('--budget-optimizer', type=str, choices=['coordinate', 'sim-anneal'], required=True)
    args = parser.parse_args()

    for model_path in args.model_paths:
        print('Starting model at {0}'.format(model_path))

        # Create the adaptive model
        controller = Controller(model_path=model_path,
                                dataset_folder=args.dataset_folder,
                                share_model=False,
                                precision=args.precision,
                                budgets=args.budgets,
                                trials=args.trials,
                                power=POWER,
                                budget_optimizer_type=args.budget_optimizer,
                                patience=args.patience,
                                max_iter=args.max_iter,
                                min_iter=args.min_iter)
        
        # Fit the model on the validation set
        controller.fit(series=DataSeries.VALID)
        controller.save()

        # print('Validation Accuracy: {0:.5f}'.format(controller.score(series=DataSeries.VALID))) 
        # print('Test Accuracy: {0:.5f}'.format(controller.score(series=DataSeries.TEST)))
