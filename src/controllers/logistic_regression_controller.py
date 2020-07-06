import numpy as np
import os.path
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from typing import List, Optional, Tuple

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from threshold_optimization.optimize_thresholds import get_serialized_info
from utils.rnn_utils import get_logits_name, get_states_name
from utils.np_utils import index_of, round_to_precision
from utils.constants import OUTPUT, BIG_NUMBER, SMALL_NUMBER
from utils.file_utils import save_pickle_gz, read_pickle_gz, extract_model_name


POWER = np.array([24.085, 32.776, 37.897, 43.952, 48.833, 50.489, 54.710, 57.692, 59.212, 59.251])
VIOLATION_FACTOR = 0.01
INCORRECT_FACTOR = 0.01
UNDERSHOOT_FACTOR = 0.005
CONTROLLER_PATH = 'model-logistic-controller-{0}.pkl.gz'
MIN_IMPROVEMENT = 0.001
ANNEAL_FACTOR = 0.999
MARGIN = 1000
MIN_INIT = 0.8
MAX_INIT = 1.0
C = 100
NOISE = 0.01


def fetch_model_states(model: AdaptiveModel, dataset: Dataset, series: DataSeries):
    logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]
    state_ops = [get_states_name(i) for i in range(model.num_outputs)]

    data_generator = dataset.minibatch_generator(series=series,
                                                 batch_size=model.hypers.batch_size,
                                                 metadata=model.metadata,
                                                 should_shuffle=False)
    # Lists to keep track of model results
    labels: List[np.ndarray] = []
    states: List[np.ndarray] = []
    level_predictions: List[np.ndarray] = []
    level_logits: List[np.ndarray] = []

    for batch_num, batch in enumerate(data_generator):
        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False)
        model_results = model.execute(feed_dict, logit_ops + state_ops)

        first_states = np.concatenate([np.expand_dims(np.squeeze(model_results[op][0]), axis=1) for op in state_ops], axis=1)
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

    states = np.concatenate(states, axis=0)
    level_predictions = np.concatenate(level_predictions, axis=0)
    labels = np.concatenate(labels, axis=0).reshape(-1, 1)
    level_logits = np.concatenate(level_logits, axis=0)

    y = (level_predictions == labels).astype(float)
    print('Level Accuracy: {0}'.format(np.average(y, axis=0)))

    return states, y, level_logits, labels


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


def predictions_for_levels(model_predictions: np.ndarray, levels: np.ndarray, batch_idx: np.ndarray) -> np.ndarray:
    preds_per_sample: List[np.ndarray] = []
    for i in range(levels.shape[0]):
        level_pred = np.squeeze(model_predictions[batch_idx, levels[i, :]])
        preds_per_sample.append(level_pred)

    preds_per_sample = np.vstack(preds_per_sample)  # [S, B]
    return preds_per_sample


def boosted_predictions(model_predictions: np.ndarray, levels: np.ndarray, error_rates: np.ndarray) -> np.ndarray:
    """
    model_predictions: [B, L, C]
    levels: [S, B]
    error_rates: [L]
    """

    level_idx = np.arange(model_predictions.shape[1]).reshape(1, 1, -1)  # [1, 1, L]
    boosted_coef = (0.5 * np.log(error_rates / ((1.0 - error_rates) + SMALL_NUMBER))).reshape(1, 1, -1)  # [1, 1, L]

    level_mask = (np.expand_dims(levels, axis=-1) >= level_idx).astype(float)  # [S, B, L]
    
    weighted_logits = np.expand_dims(model_predictions, axis=0) * np.expand_dims(level_mask * boosted_coef, axis=-1)  # [S, B, L, C]
    print(weighted_logits.shape)

    weighted_logits = np.sum(weighted_logits, axis=-2)  # [S, B, C]
    print(weighted_logits.shape)

    weighted_predictions = np.argmax(weighted_logits, axis=-1)  # [S, B]
    print(weighted_predictions.shape)

    return weighted_predictions


### Budget optimizer classes ###

class BudgetOptimizer:
    
    def __init__(self, num_levels: int, budgets: np.ndarray, precision: int, trials: int, max_iter: int, patience: int):
        self._num_levels = num_levels
        self._num_budgets = budgets.shape[0]
        self._budgets = budgets
        self._precision = precision
        self._trials = trials
        self._max_iter = max_iter
        self._patience = patience
        self._rand = np.random.RandomState(seed=42)
        self._thresholds = None

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
        approx_power = np.sum(normalized_level_counts * POWER, axis=-1).astype(float)  # [S]
        return approx_power


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
            approx_power = self.get_approx_power(levels=levels)
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

    def fitness_function(self, thresholds: np.ndarray, network_results: np.ndarray, clf_predictions: np.ndarray, batch_size: int, violation_factor: float):
        # Compute the number of levels to execute
        levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=thresholds)  # [B]

        # Compute the approximate power
        approx_power = self.get_approx_power(levels=levels)
        dual_term = approx_power - self._budgets  # [S]
        dual_penalty = np.where(dual_term > 0, violation_factor * dual_term, 0.0)
        # dual_penalty = violation_factor * np.square(dual_term)

        # Compute the accuracy
        batch_idx = np.arange(start=0, stop=batch_size)  # [B]
        correct_per_level = predictions_for_levels(model_predictions=network_results, levels=levels, batch_idx=batch_idx)

        accuracy = np.average(correct_per_level, axis=-1)  # [S]

        # Find the samples in which all levels are wrong. In these cases, we want to fail early. Thus,
        # we penalize the number of levels executed in these cases.
        #sample_incorrect = np.equal(np.min(network_results, axis=-1), 0)  # [B]. 1 when all levels are wrong
        #incorrect_sample_levels = levels * sample_incorrect
        #avg_levels_incorrect = np.sum(incorrect_sample_levels) / np.sum(sample_incorrect)
        #incorrect_penalty = INCORRECT_FACTOR * avg_levels_incorrect
        # print('Incorrect Penalty: {0}, Avg Levels Incorrect: {1}'.format(incorrect_penalty, avg_levels_incorrect))

        # var_penalty = 0.01 * np.std(levels)

        return -accuracy + dual_penalty, approx_power

    def fit(self, network_results: np.ndarray, clf_predictions: np.ndarray):
        # Split into training and validation sets
        sample_idx = np.arange(network_results.shape[0])  # [B]
        self._rand.shuffle(sample_idx)
        
        split_point = int(len(sample_idx) * 0.85)
        # train_idx = sample_idx[:split_point]
        # valid_idx = sample_idx[split_point:]

        train_idx = np.arange(network_results.shape[0])
        valid_idx = np.arange(network_results.shape[0])

        train_network_results, train_clf_predictions = network_results[train_idx], clf_predictions[train_idx]
        valid_network_results, valid_clf_predictions = network_results[valid_idx], clf_predictions[valid_idx]

        best_thresholds = np.ones(shape=(self._num_budgets, self._num_levels))
        best_fitness = np.ones(shape=(self._num_budgets, 1), dtype=float)

        # Reshape the validation arrays
        valid_clf_predictions = np.expand_dims(valid_clf_predictions, axis=0)  # [1, B, L]

        for t in range(self._trials):
            print('===== Starting Trial {0} ====='.format(t))

            init_thresholds = np.random.uniform(low=MIN_INIT, high=MAX_INIT, size=(self._num_budgets, self._num_levels))
            init_thresholds = round_to_precision(init_thresholds, self._precision)

            thresholds = self.fit_single(network_results=train_network_results,
                                         clf_predictions=train_clf_predictions,
                                         init_thresholds=init_thresholds)

            # Compute the fitness
            fitness, _ = self.fitness_function(thresholds=thresholds,
                                               network_results=valid_network_results,
                                               clf_predictions=valid_clf_predictions,
                                               batch_size=valid_clf_predictions.shape[1],
                                               violation_factor=0.0)
            fitness = np.expand_dims(fitness, axis=1)

            best_thresholds = np.where(fitness < best_fitness, thresholds, best_thresholds)
            best_fitness = np.where(fitness < best_fitness, fitness, best_fitness)
            print('Completed Trial {0}. Best Fitness: {1}'.format(t, best_fitness))

        levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=best_thresholds)
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

        self._thresholds = best_thresholds
        return best_thresholds

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
        margin = MIN_IMPROVEMENT

        best_fitness = np.ones(shape=(self._num_budgets,), dtype=float)
        best_power = np.zeros_like(best_fitness)

        violation_factor = VIOLATION_FACTOR

        for i in range(self._max_iter):

            # Select a random level to run
            level = self._rand.randint(low=0, high=self._num_levels - 1)

            # [S] array of threshold values
            best_t = np.copy(thresholds[:, level])  # The 'best' are the previous thresholds at this level
            
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
                                                              violation_factor=violation_factor)

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

                best_t = np.where(fitness <= best_fitness, candidate_values, best_t)
                best_power = np.where(fitness <= best_fitness, approx_power, best_power)
                best_fitness = np.where(fitness <= best_fitness, fitness, best_fitness)

            thresholds[:, level] = best_t  # Set the best thresholds
            print('Completed Iteration: {0}: level {1}'.format(i, level))
            print('\tBest Fitness: {0}'.format(-1 * best_fitness))
            print('\tApprox Power: {0}'.format(best_power))
            print('\tThresholds: {0}'.format(thresholds))

            if (np.isclose(thresholds, prev_thresholds)).all():
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0

            if early_stopping_counter >= self._patience:
                print('Converged.')
                break

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
                 budget_optimizer_type: str,
                 patience: int,
                 max_iter: int):
        self._model_path = model_path
        self._dataset_folder = dataset_folder

        # Load the model and dataset
        model, dataset, _ = get_serialized_info(model_path, dataset_folder=dataset_folder)

        self._model = model
        self._dataset = dataset
        self._is_fitted = False
        self._share_model = share_model
        self._num_levels = model.num_outputs

        # Predicts whether all levels will get the sample incorrect
        self._incorrect_clf = LogisticRegression(C=C, max_iter=500)

        if share_model:
            # self._clf = LogisticRegression(C=C, max_iter=500)
            # self._clf = AdaBoostClassifier(random_state=42)
            self._clf = RandomForestClassifier(max_depth=5, random_state=32)
        else:
            self._clf = [LogisticRegression(C=C, max_iter=500) for _ in range(self._num_levels)]

        self._budgets = np.array(budgets)
        self._num_budgets = len(self._budgets)
        self._precision = precision
        self._trials = trials
        self._thresholds = None
        self._patience = patience
        self._max_iter = max_iter

        # Create the budget optimizer
        self._budget_optimizer_type = budget_optimizer_type.lower()
        if self._budget_optimizer_type == 'coordinate':
            self._budget_optimizer = CoordinateOptimizer(num_levels=self._num_levels,
                                                         budgets=self._budgets,
                                                         precision=self._precision,
                                                         trials=self._trials,
                                                         patience=patience,
                                                         max_iter=max_iter)
        elif self._budget_optimizer_type == 'sim-anneal':
            self._budget_optimizer = SimulatedAnnealingOptimizer(num_levels=self._num_levels,
                                                                 budgets=self._budgets,
                                                                 precision=self._precision,
                                                                 trials=self._trials,
                                                                 temp=0.1,
                                                                 anneal_rate=0.95,
                                                                 patience=patience,
                                                                 max_iter=max_iter)
        else:
            raise ValueError('Unknown budget optimizer: {0}'.format(budget_optimizer_type))

    def fit(self, series: DataSeries):
        X_train, y_train, train_logits, train_labels = fetch_model_states(self._model, self._dataset, series=series)
        X_test, y_test, test_logits, test_labels = fetch_model_states(self._model, self._dataset, series=DataSeries.TEST)

        # Fit the model to predict when labels will be incorrect (for all levels)
        #incorrect_labels = (1.0 - np.max(y_train, axis=-1, keepdims=True))  # [B, 1]
        #y_incorrect = np.repeat(incorrect_labels, repeats=y_train.shape[-1], axis=-1)  # [B, L]

        #self._incorrect_clf.fit(X_train.reshape(-1, X_train.shape[-1]), y_incorrect.reshape(-1))
        #inc_acc = self._incorrect_clf.score(X_train.reshape(-1, X_train.shape[-1]), y_incorrect.reshape(-1))
        #probs = self._incorrect_clf.predict_proba(X_train.reshape(-1, X_train.shape[-1]))[:, 1]

        # Fit the logistic regression model
        if self._share_model:
            X_train = X_train.reshape(-1, X_train.shape[-1])

            self._clf.fit(X_train, y_train.reshape(-1))
            clf_predictions = self._clf.predict_proba(X_train)[:, 1]

            clf_predictions = clf_predictions.reshape(-1, self._num_levels)

            test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            test_clf_predictions = self._clf.predict_proba(test_reshaped)[:, 1]
            test_clf_predictions = test_clf_predictions.reshape(-1, self._num_levels)
        else:
            clf_predictions: List[np.ndarray] = []
            test_clf_predictions: List[np.ndarray] = []
            for level in range(self._num_levels):
                X_input = X_train[:, level, :]

                self._clf[level].fit(X_input, y_train[:, level])
                clf_predictions.append(self._clf[level].predict_proba(X_input)[:, 1])

                X_test_input = X_test[:, level, :]
                test_clf_predictions.append(self._clf[level].predict_proba(X_test_input)[:, 1])

                # print(np.average(y_train[:, level]))
                # print(self._clf[level].score(X_input, y_train[:, level]))

            clf_predictions = np.transpose(np.array(clf_predictions))  # [B, L]
            test_clf_predictions = np.transpose(np.array(test_clf_predictions))

        #print('TRAINING DISTRIBUTIONS')
        #zero_predictions = (1.0 - y_train) * clf_predictions
        #one_predictions = y_train * clf_predictions
        #zero_avg = np.sum(zero_predictions, axis=0) / np.sum((1.0 - y_train), axis=0, keepdims=True)
        #one_avg = np.sum(one_predictions, axis=0) / np.sum(y_train, axis=0, keepdims=True)

        #print('Zero Stats: Avg -> {0}'.format(zero_avg))
        #print('One Stats: Avg -> {0}'.format(one_avg))

        #print('TESTING DISTRIBUTIONS')
        #zero_predictions = (1.0 - y_test) * test_clf_predictions
        #one_predictions = y_test * test_clf_predictions
        #zero_avg = np.sum(zero_predictions, axis=0) / np.sum((1.0 - y_test), axis=0, keepdims=True)
        #one_avg = np.sum(one_predictions, axis=0) / np.sum(y_test, axis=0, keepdims=True)

        #print('Zero Stats: Avg -> {0}'.format(zero_avg))
        #print('One Stats: Avg -> {0}'.format(one_avg))

        #print(clf_predictions.shape)
        #print(test_clf_predictions.shape)

        # Fit the thresholds
        self._thresholds = self._budget_optimizer.fit(network_results=y_train, clf_predictions=clf_predictions)

        # Evaluate the model optimizer
        print('======')
        train_acc = self._budget_optimizer.evaluate(network_results=y_train, clf_predictions=clf_predictions)
        test_acc = self._budget_optimizer.evaluate(network_results=y_test, clf_predictions=test_clf_predictions)

        print('Train Accuracy: {0}'.format(train_acc))
        print('Test Accuracy: {0}'.format(test_acc))

        print('=====')

        self._is_fitted = True

    def score(self, series: DataSeries) -> np.ndarray:
        assert self._is_fitted, 'Model is not fitted'
        X, y, _, _ = fetch_model_states(self._model, self._dataset, series=series)

        if self._share_model:
            X = X.reshape(-1, X.shape[-1])
            y = y.reshape(-1)

            accuracy = self._clf.score(X, y)
        else:
            total_accuracy = 0.0
            for level in range(self._model.num_outputs):
                total_accuracy += self._clf[level].score(X[:, level, :], y[:, level])

            accuracy = total_accuracy / self._model.num_outputs

        return accuracy

    def predict_sample(self, states: List[np.ndarray], budget: int) -> int:
        """
        Predicts the number of levels given the list of hidden states. The states are assumed to be in order.

        Args:
            states: A list of length num_levels containing the first hidden state from each model level.
            budget: The budget to perform inference under. This controls the employed thresholds.
        Returns:
            The number of levels to execute.
        """
        assert self._is_fitted, 'Model is not fitted'

        budget_idx = index_of(self._budgets, value=budget)
        assert budget_idx >= 0, 'Could not find values for budget {0}'.format(budget)

        # Get thresholds for this budget
        thresholds = self._thresholds[budget_idx]

        for level, state in enumerate(states):
            # Compute the logistic model predictions. Choice of classifier depends on the model sharing choice.
            state = state.reshape(1, -1)
            if self._share_model:
                logistic_prob = self._clf.predict_proba(state)[0, 1]
            else:
                logistic_prob = self._clf[level].predict_proba(state)[0, 1]

            # Return early if we find a probability larger than the corresponding threshold.
            if thresholds[level] < logistic_prob:
                return level

        # By default, we return the top level
        return self._num_levels

    def predict_levels(self, series: DataSeries, budget: float) -> np.ndarray:
        assert self._is_fitted, 'Model is not fitted'

        budget_idx = index_of(self._budgets, value=budget)
        assert budget_idx >= 0, 'Could not find values for budget {0}'.format(budget)

        X, _, _ = fetch_model_states(self._model, self._dataset, series=series)

        if self._share_model:
            X = X.reshape(-1, X.shape[-1])
            clf_predictions = self._clf.predict_proba(X)[:, 1]
            clf_predictions = clf_predictions.reshape(-1, self._model.num_outputs)  # [B, L]
        else:
            clf_predictions: List[np.ndarray] = []
            for level in range(self._model.num_outputs):
                X_input = X[:, level, :]
                clf_predictions.append(self._clf[level].predict_proba(X_input)[:, 1])

            clf_predictions = np.transpose(np.array(clf_predictions))  # [B, L]

        clf_predictions = np.expand_dims(clf_predictions, axis=0)  # [1, B, L]

        levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=self._thresholds)
        budget_levels = levels[budget_idx]
        return budget_levels.astype(int)

    def as_dict(self):
        return {
            'clf': self._clf,
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
            'budget_optimizer_type': self._budget_optimizer_type
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
                                max_iter=serialized_info.get('max_iter', 100))

        # Set remaining fields
        controller._clf = serialized_info['clf']
        controller._thresholds = serialized_info['thresholds']
        controller._is_fitted = serialized_info['is_fitted']

        return controller


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-paths', type=str, nargs='+')
    parser.add_argument('--dataset-folder', type=str)
    parser.add_argument('--budgets', type=float, nargs='+')
    parser.add_argument('--precision', type=int, required=True)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--budget-optimizer', type=str, choices=['coordinate', 'sim-anneal'])
    args = parser.parse_args()

    for model_path in args.model_paths:
        print('Starting model at {0}'.format(model_path))

        # Create the adaptive model
        controller = Controller(model_path=model_path,
                                dataset_folder=args.dataset_folder,
                                share_model=True,
                                precision=args.precision,
                                budgets=args.budgets,
                                trials=args.trials,
                                budget_optimizer_type=args.budget_optimizer,
                                patience=args.patience,
                                max_iter=args.max_iter)
        
        # Fit the model on the validation set
        controller.fit(series=DataSeries.VALID)
        controller.save()

        print('Validation Accuracy: {0:.5f}'.format(controller.score(series=DataSeries.VALID)))
        print('Test Accuracy: {0:.5f}'.format(controller.score(series=DataSeries.TEST)))
