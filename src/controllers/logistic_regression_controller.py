import numpy as np
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import List

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from threshold_optimization.optimize_thresholds import get_serialized_info
from utils.rnn_utils import get_logits_name, get_states_name
from utils.constants import OUTPUT, BIG_NUMBER, SMALL_NUMBER


POWER = np.array([24.085, 32.776, 37.897, 43.952, 48.833, 50.489, 54.710, 57.692, 59.212, 59.251])
VIOLATION_FACTOR = 100


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

    for batch_num, batch in enumerate(data_generator):
        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False)
        model_results = model.execute(feed_dict, logit_ops + state_ops)

        first_states = np.concatenate([np.expand_dims(np.squeeze(model_results[op][0]), axis=1) for op in state_ops], axis=1)
        states.append(first_states)

        # Concatenate logits into a [B, L, C] array (logit_ops is already ordered by level).
        # For reference, L is the number of levels and C is the number of classes
        logits_concat = np.concatenate([np.expand_dims(model_results[op], axis=1) for op in logit_ops], axis=1)

        # Compute the predictions for each level
        level_pred = np.argmax(logits_concat, axis=-1)  # [B, L]
        level_predictions.append(level_pred)
        
        true_values = np.squeeze(batch[OUTPUT])
        labels.append(true_values)

    states = np.concatenate(states, axis=0)
    level_predictions = np.concatenate(level_predictions, axis=0)
    labels = np.concatenate(labels, axis=0).reshape(-1, 1)

    y = (level_predictions == labels).astype(float)

    return states, y, level_predictions


def levels_to_execute(logistic_probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    # Compute the predictions based on this threshold setting. The level predictions are a 0/1
    # array which is 0 when we should NOT use this level and 1 when we should
    expanded_thresholds = np.expand_dims(thresholds, axis=1)  # [S, 1, L]
    level_predictions = (logistic_probs < expanded_thresholds).astype(int)  # [S, B, L]

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


class Controller:

    def __init__(self, model: AdaptiveModel, dataset: Dataset, share_model: bool, precision: int, budgets: List[float], trials: int):
        self._model = model
        self._dataset = dataset
        self._clfs = [LogisticRegression(C=1.0, max_iter=500) for _ in range(model.num_outputs)]
        self._is_fitted = False
        self._share_model = share_model

        if share_model:
            self._clf = LogisticRegression(C=1.0, max_iter=500)
        else:
            self._clf = [LogisticRegression(C=1.0, max_iter=500) for _ in range(model.num_outputs)]

        self._budgets = np.array(budgets)
        self._precision = precision
        self._trials = trials
    
    def fit(self, series: DataSeries):
        X_train, y_train, model_predictions = fetch_model_states(self._model, self._dataset, series=series)

        if self._share_model:
            X_train = X_train.reshape(-1, X_train.shape[-1])

            self._clf.fit(X_train, y_train.reshape(-1))
            clf_predictions = self._clf.predict_proba(X_train)[:, 1]

            clf_predictions = clf_predictions.reshape(-1, self._model.num_outputs)
        else:
            clf_predictions: List[np.ndarray] = []
            for level in range(self._model.num_outputs):
                X_input = X_train[:, level, :]

                self._clf[level].fit(X_input, y_train[:, level])
                clf_predictions.append(self._clf[level].predict_proba(X_input)[:, 1])

            clf_predictions = np.transpose(np.array(clf_predictions))  # [B, L]

        clf_predictions = np.expand_dims(clf_predictions, axis=0)  # [1, B, L]

        # Optimize the level thresholds given the budget and fixed point precision
        thresholds = np.zeros(shape=(len(self._budgets), self._model.num_outputs))
        thresholds[:, -1] = 0

        # The number 1 in fixed point representation
        fp_one = 1 << self._precision

        # Array of level indices
        level_idx = np.arange(start=0, stop=self._model.num_outputs).reshape(1, 1, -1)  # [1, 1, L]
        batch_idx = np.arange(start=0, stop=clf_predictions.shape[1])  # [B]

        prev_thresholds = np.copy(thresholds)
        for trial in range(self._trials):

            print('===== Starting Trial {0} ====='.format(trial))

            for level in reversed(range(self._model.num_outputs - 1)):

                best_t = np.zeros(shape=(len(self._budgets), ), dtype=float)
                best_fitness = np.full(shape=(len(self._budgets), ), fill_value=BIG_NUMBER)

                for t in range(0, fp_one + 1):
                    
                    # Compute the predictions using the threshold on the logistic regression model
                    thresholds[:, level] = t / fp_one

                    levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=thresholds)

                    # Compute the fitness function
                    level_counts = np.vstack([np.bincount(levels[i, :], minlength=self._model.num_outputs) for i in range(levels.shape[0])])  # [S, L]
                    normalized_level_counts = level_counts / np.sum(level_counts, axis=-1, keepdims=True)  # [S, L]
                    approx_power = np.sum(normalized_level_counts * POWER, axis=-1).astype(float)  # [S]
                    dual_penalty = approx_power - self._budgets  # [S]

                    correct_per_level = predictions_for_levels(model_predictions=y_train, levels=levels, batch_idx=batch_idx)
                    accuracy = np.average(correct_per_level, axis=-1)  # [S]
                    fitness = -accuracy + VIOLATION_FACTOR * np.clip(dual_penalty, a_min=0.0, a_max=None) # [S]

                    best_t = np.where(fitness < best_fitness, t / fp_one, best_t)
                    best_fitness = np.minimum(best_fitness, fitness)

                thresholds[:, level] = best_t  # Set the best thresholds
                print('Completed Level {0}'.format(level))
                print('\tBest Fitness: {0}'.format(best_fitness))

            if (np.isclose(thresholds, prev_thresholds)).all():
                print('Converged.')
                break

            prev_thresholds = np.copy(thresholds)
                
        self._thresholds = thresholds
        self._is_fitted = True

    def score(self, series: DataSeries) -> np.ndarray:
        assert self._is_fitted, 'Model is not fitted'
        X, y, _ = fetch_model_states(self._model, self._dataset, series=series)

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

    def predict_levels(self, series: DataSeries, budget: int) -> np.ndarray:
        assert self._is_fitted, 'Model is not fitted'

        budget_idx = -1
        for idx, b in enumerate(self._budgets):
            if abs(budget - b) < SMALL_NUMBER:
                budget_idx = idx
                break

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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str)
    args = parser.parse_args()

    # Create the adaptive model
    model, dataset, test_log = get_serialized_info(args.model_path, dataset_folder=args.dataset_folder)

    controller = Controller(model=model, dataset=dataset, share_model=True, precision=10, budgets=[47], trials=10)
    controller.fit(series=DataSeries.VALID)

    # print('Train Accuracy: {0:.5f}'.format(controller.score(series=DataSeries.VALID)))
    # print('Test Accuracy: {0:.5f}'.format(controller.score(series=DataSeries.TEST)))

    levels = controller.predict_levels(series=DataSeries.TEST, budget=47)
    print(np.average(levels))
    print(np.max(levels))
