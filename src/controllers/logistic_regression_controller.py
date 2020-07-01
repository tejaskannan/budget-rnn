import numpy as np
import os.path
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from typing import List, Optional

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from threshold_optimization.optimize_thresholds import get_serialized_info
from utils.rnn_utils import get_logits_name, get_states_name
from utils.np_utils import index_of
from utils.constants import OUTPUT, BIG_NUMBER, SMALL_NUMBER
from utils.file_utils import save_pickle_gz, read_pickle_gz, extract_model_name


POWER = np.array([24.085, 32.776, 37.897, 43.952, 48.833, 50.489, 54.710, 57.692, 59.212, 59.251])
VIOLATION_FACTOR = 0.01
UNDERSHOOT_FACTOR = 0.0
CONTROLLER_PATH = 'model-logistic-controller-{0}.pkl.gz'
MIN_IMPROVEMENT = 0.001
ANNEAL_FACTOR = 0.999


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
    print('Level Accuracy: {0}'.format(np.average(y, axis=0)))

    return states, y, level_predictions


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


### Budget optimizer classes ###

class BudgetOptimizer:
    
    def __init__(self, num_levels: int, budgets: np.ndarray, precision: int, trials: int):
        self._num_levels = num_levels
        self._num_budgets = budgets.shape[0]
        self._budgets = budgets
        self._precision = precision
        self._trials = trials
        self._rand = np.random.RandomState(seed=42)

    def fit(self, clf_predictions: np.ndarray, patience: int):
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


class CoordinateOptimizer(BudgetOptimizer):

    def fit(self, network_results: np.ndarray, clf_predictions: np.ndarray, patience: int):
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

        # Initialize thresholds, [S, L] array
        thresholds = np.ones(shape=(self._num_budgets, self._num_levels))
        thresholds[:, -1] = 0

        # The number 1 in fixed point representation
        fp_one = 1 << self._precision

        # Array of level indices
        level_idx = np.arange(start=0, stop=self._num_levels).reshape(1, 1, -1)  # [1, 1, L]
        batch_idx = np.arange(start=0, stop=clf_predictions.shape[1])  # [B]

        # Variable for convergence
        early_stopping_counter = 0
        prev_thresholds = np.copy(thresholds)
        margin = MIN_IMPROVEMENT

        best_fitness = np.ones(shape=(self._num_budgets,), dtype=float)
        best_power = np.zeros_like(best_fitness)

        for trial in range(self._trials):

            print('===== Starting Trial {0} ====='.format(trial))

            for _ in range(self._num_levels - 1):

                # Select a random level to run
                level = self._rand.randint(low=0, high=self._num_levels - 1)

               # best_t = np.ones(shape=(self._num_budgets,), dtype=float)
               # best_fitness = np.zeros(shape=(self._num_budgets,), dtype=float)

                best_t = np.copy(thresholds[:, level])  # The 'best' are the previous thresholds at this level
                # best_power = np.zeros_like(best_t)

                for t in reversed(range(0, fp_one + 1)):
                    
                    # Compute the predictions using the threshold on the logistic regression model
                    thresholds[:, level] = t / fp_one

                    # [S, B]
                    levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=thresholds)

                    # Compute the approximate power and accuracy
                    approx_power = self.get_approx_power(levels=levels)
                    dual_term = approx_power - self._budgets  # [S]
                    dual_penalty = np.where(dual_term > 0, VIOLATION_FACTOR * dual_term, -UNDERSHOOT_FACTOR * dual_term)

                    correct_per_level = predictions_for_levels(model_predictions=network_results, levels=levels, batch_idx=batch_idx)
                    accuracy = np.average(correct_per_level, axis=-1)  # [S]

                    # Regularization term to avoid large changes
                    # threshold_diff = np.abs(thresholds[:, level] - prev_thresholds[:, level])

                    # Compute the fitness
                    # fitness = -accuracy + VIOLATION_FACTOR * np.clip(dual_penalty, a_min=0.0, a_max=None) + 0.01 * threshold_diff # [S]
                    fitness = -accuracy + dual_penalty

                    best_t = np.where(fitness < best_fitness, t / fp_one, best_t)
                    best_power = np.where(fitness < best_fitness, approx_power, best_power)
                    best_fitness = np.where(fitness < best_fitness, fitness, best_fitness)

                thresholds[:, level] = best_t  # Set the best thresholds
                print('Completed Level {0}'.format(level))
                print('\tBest Fitness: {0}'.format(-1 * best_fitness))
                print('\tApprox Power: {0}'.format(best_power))
                # print('\tThresholds: {0}'.format(thresholds))

            if (np.isclose(thresholds, prev_thresholds)).all():
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0

            if early_stopping_counter >= patience:
                print('Converged.')
                break

            prev_thresholds = np.copy(thresholds)

        return thresholds


### Model Controllers ###

class Controller:

    def __init__(self, model_path: str, dataset_folder: str, share_model: bool, precision: int, budgets: List[float], trials: int, budget_optimizer_type: str):
        self._model_path = model_path
        self._dataset_folder = dataset_folder

        # Load the model and dataset
        model, dataset, _ = get_serialized_info(model_path, dataset_folder=dataset_folder)

        self._model = model
        self._dataset = dataset
        self._is_fitted = False
        self._share_model = share_model
        self._num_levels = model.num_outputs

        if share_model:
            self._clf = LogisticRegression(C=1.0, max_iter=500)
        else:
            self._clf = [LogisticRegression(C=1.0, max_iter=500) for _ in range(self._num_levels)]

        self._budgets = np.array(budgets)
        self._num_budgets = len(self._budgets)
        self._precision = precision
        self._trials = trials
        self._thresholds = None

        # Create the budget optimizer
        self._budget_optimizer_type = budget_optimizer_type.lower()
        if self._budget_optimizer_type == 'coordinate':
            self._budget_optimizer = CoordinateOptimizer(num_levels=self._num_levels,
                                                         budgets=self._budgets,
                                                         precision=self._precision,
                                                         trials=self._trials)
        else:
            raise ValueError('Unknown budget optimizer: {0}'.format(budget_optimizer_type))

    def fit(self, series: DataSeries, patience: int):
        X_train, y_train, model_predictions = fetch_model_states(self._model, self._dataset, series=series)

        # Fit the logistic regression model
        if self._share_model:
            X_train = X_train.reshape(-1, X_train.shape[-1])

            self._clf.fit(X_train, y_train.reshape(-1))
            clf_predictions = self._clf.predict_proba(X_train)[:, 1]

            clf_predictions = clf_predictions.reshape(-1, self._num_levels)
        else:
            clf_predictions: List[np.ndarray] = []
            for level in range(self._num_levels):
                X_input = X_train[:, level, :]

                self._clf[level].fit(X_input, y_train[:, level])
                clf_predictions.append(self._clf[level].predict_proba(X_input)[:, 1])

            clf_predictions = np.transpose(np.array(clf_predictions))  # [B, L]

        # Fit the thresholds
        self._thresholds = self._budget_optimizer.fit(network_results=y_train, clf_predictions=clf_predictions, patience=patience)
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
                                budget_optimizer_type=serialized_info['budget_optimizer_type'])

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
    parser.add_argument('--trials', type=int, default=15)
    parser.add_argument('--patience', type=int, default=3)
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
                                budget_optimizer_type=args.budget_optimizer)
        
        # Fit the model on the validation set
        controller.fit(series=DataSeries.VALID, patience=args.patience)
        controller.save()

        print('Train Accuracy: {0:.5f}'.format(controller.score(series=DataSeries.VALID)))
        print('Test Accuracy: {0:.5f}'.format(controller.score(series=DataSeries.TEST)))
