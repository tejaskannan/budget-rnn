import numpy as np
import os.path
from argparse import ArgumentParser
from typing import List, Optional, Tuple

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from threshold_optimization.optimize_thresholds import get_serialized_info
from utils.rnn_utils import get_logits_name, get_states_name, AdaptiveModelType, get_input_name, is_cascade, is_sample, get_stop_output_name
from utils.np_utils import index_of, round_to_precision
from utils.constants import OUTPUT, BIG_NUMBER, SMALL_NUMBER, INPUTS, SEQ_LENGTH, DROPOUT_KEEP_RATE, SEQ_LENGTH
from utils.file_utils import save_pickle_gz, read_pickle_gz, extract_model_name
from controllers.distribution_prior import DistributionPrior
from controllers.power_utils import get_avg_power_multiple, get_avg_power, get_weighted_avg_power
from controllers.controller_utils import execute_adaptive_model


VIOLATION_FACTOR = 1.0
UNDERSHOOT_FACTOR = 1.0
CONTROLLER_PATH = 'model-controller-{0}.pkl.gz'
MARGIN = 1000
MIN_INIT = 0.8
MAX_INIT = 1.0
FACTOR_START = 1e-4



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


def fit_anneal_rate(start_value: float, end_value: float, steps: int):
    return np.exp((1.0 / steps) * np.log(max(end_value, SMALL_NUMBER) / max(start_value, SMALL_NUMBER)))


### Budget optimizer class ###

class BudgetOptimizer:
    
    def __init__(self, num_levels: int, budgets: np.ndarray, seq_length: int, precision: int, trials: int, max_iter: int, min_iter: int, patience: int):
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
        self._seq_length = seq_length

    def evaluate(self, network_predictions: np.ndarray, clf_predictions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def fitness_function(self, thresholds: np.ndarray, network_results: np.ndarray, clf_predictions: np.ndarray, batch_size: int, violation_factor: float, undershoot_factor: float):
        # Compute the number of levels to execute
        levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=thresholds)  # [S, B]

        # Compute the approximate power
        power_multiplier = int(self._seq_length / self._num_levels)
        approx_power = np.vstack([get_avg_power_multiple(levels[idx] + 1, self._seq_length, power_multiplier) for idx in range(self._num_budgets)])  # [S, 1]
        approx_power = np.squeeze(approx_power, axis=-1)  # [S]

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

            # Compute the fitness, [S]
            fitness, _ = self.fitness_function(thresholds=thresholds,
                                               network_results=network_results,
                                               clf_predictions=valid_clf_predictions,
                                               batch_size=valid_clf_predictions.shape[1],
                                               violation_factor=VIOLATION_FACTOR,
                                               undershoot_factor=UNDERSHOOT_FACTOR)
            fitness = np.expand_dims(fitness, axis=-1)  # [S, 1]

            # Set the thresholds using the best seen fitness so far
            best_thresholds = np.where(fitness < best_fitness, thresholds, best_thresholds)
            best_fitness = np.where(fitness < best_fitness, fitness, best_fitness)
            print('Completed Trial {0}. Best Fitness: {1}'.format(t, best_fitness))

        levels = levels_to_execute(logistic_probs=clf_predictions, thresholds=best_thresholds)
        level_counts = np.vstack([np.bincount(levels[i], minlength=self._num_levels) for i in range(self._num_budgets)])  # [S, L]
        avg_level_counts = level_counts / np.sum(level_counts, axis=-1, keepdims=True)

        self._thresholds = best_thresholds
        return best_thresholds, avg_level_counts

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

        # Initialize penalty parameters
        violation_factor = FACTOR_START
        entropy_factor = FACTOR_START
        undershoot_factor = FACTOR_START

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

            # Variables for tie-breaking
            steps = np.zeros_like(best_fitness)
            prev_level_fitness = np.ones_like(best_fitness)
            prev_level_approx_power = np.zeros_like(best_power)
            current_thresholds = np.zeros_like(best_t)  # [S]

            for offset in range(MARGIN):

                # Compute the predictions using the threshold on the logistic regression model
                candidate_values = np.minimum((start_values + offset) / fp_one, 1)
                thresholds[:, level] = candidate_values

                # Compute the fitness
                fitness, avg_power = self.fitness_function(thresholds=thresholds,
                                                           network_results=network_results,
                                                           clf_predictions=clf_predictions,
                                                           batch_size=clf_predictions.shape[1],
                                                           violation_factor=violation_factor,
                                                           undershoot_factor=undershoot_factor)

                best_t = np.where(fitness < best_fitness, candidate_values, best_t)
                best_power = np.where(fitness < best_fitness, avg_power, best_power)
                best_fitness = np.where(fitness < best_fitness, fitness, best_fitness)

            thresholds[:, level] = best_t  # Set the best thresholds
            print('Completed Iteration: {0}: level {1}'.format(i, level))
            print('\tBest Fitness: {0}'.format(-1 * best_fitness))
            print('\tApprox Power: {0}'.format(best_power))

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
        self._seq_length = model.metadata[SEQ_LENGTH]

        self._budgets = np.array(list(sorted(budgets)))
        self._num_budgets = len(self._budgets)
        self._precision = precision
        self._trials = trials
        self._thresholds = None
        self._patience = patience
        self._max_iter = max_iter
        self._min_iter = min_iter
        self._levels_per_label = None

        # Create the budget optimizer
        self._budget_optimizer = BudgetOptimizer(num_levels=self._num_levels,
                                                 seq_length=self._seq_length,
                                                 budgets=self._budgets,
                                                 precision=self._precision,
                                                 trials=self._trials,
                                                 patience=patience,
                                                 max_iter=max_iter,
                                                 min_iter=min_iter)

    def fit(self, series: DataSeries):
        train_results = execute_adaptive_model(self._model, self._dataset, series=series)
        test_results = execute_adaptive_model(self._model, self._dataset, series=DataSeries.TEST)

        train_correct = train_results.predictions == train_results.labels  # [N, L]
        test_correct = test_results.predictions == test_results.labels  # [M, L]

        # Fit the thresholds
        self._thresholds, self._avg_level_counts = self._budget_optimizer.fit(network_results=train_correct, clf_predictions=train_results.stop_probs)
    
        # Evaluate the model optimizer
        train_acc = self._budget_optimizer.evaluate(network_results=train_correct, clf_predictions=train_results.stop_probs)
        test_acc = self._budget_optimizer.evaluate(network_results=test_correct, clf_predictions=test_results.stop_probs)

        print('Train Accuracy: {0}'.format(train_acc))
        print('Test Accuracy: {0}'.format(test_acc))

        self._is_fitted = True

    def get_thresholds(self, budget: int) -> np.ndarray:
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
                upper_budget  = max_power

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

    def predict_sample(self, stop_probs: np.ndarray, budget: int) -> int:
        """
        Predicts the number of levels given the list of hidden states. The states are assumed to be in order.

        Args:
            stop_probs: An array of [L] stop probabilities, one for each level
            budget: The budget to perform inference under. This controls the employed thresholds.
        Returns:
            The number of levels to execute.
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

    def predict_levels(self, series: DataSeries, budget: float) -> Tuple[np.ndarray, np.ndarray]:
        assert self._is_fitted, 'Model is not fitted'

        thresholds = self.get_thresholds(budget)

        results = execute_adaptive_model(self._model, self._dataset, series=series)

        levels = levels_to_execute(logistic_probs=results.stop_probs, thresholds=np.expand_dims(thresholds, axis=0))

        batch_idx = np.arange(results.predictions.shape[0])
        predictions = predictions_for_levels(model_predictions=results.predictions,
                                             levels=levels,
                                             batch_idx=batch_idx)

        return levels[0].astype(int), predictions[0].astype(int)

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
    def load(cls, save_file: str, dataset_folder: Optional[str] = None):
        """
        Loads the controller from the given serialized file.
        """
        # Load the serialized information.
        serialized_info = read_pickle_gz(save_file)
        dataset_folder = dataset_folder if dataset_folder is not None else serialized_info['dataset_folder']

        # Initialize the new controller
        controller = Controller(model_path=serialized_info['model_path'],
                                dataset_folder=dataset_folder,
                                share_model=serialized_info['share_model'],
                                precision=serialized_info['precision'],
                                budgets=serialized_info['budgets'],
                                trials=serialized_info['trials'],
                                patience=serialized_info.get('patience', 10),
                                max_iter=serialized_info.get('max_iter', 100),
                                min_iter=serialized_info.get('min_iter', 20))

        # Set remaining fields
        controller._thresholds = serialized_info['thresholds']
        controller._avg_level_counts = serialized_info['avg_level_counts']
        controller._is_fitted = serialized_info['is_fitted']

        return controller


class FixedController(Controller):

    def __init__(self, model_index: int):
        self._model_index = model_index
    
    def fit(self, series: DataSeries):
        pass
    
    def predict_sample(self, stop_probs: np.ndarray, budget: int) -> int:
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

        self._threshold_dict: Dict[float, np.ndarray] = dict()

        # Create random state for reproducible results
        self._rand = np.random.RandomState(seed=62)
        self._is_fitted = False

    def fit(self, series: DataSeries):
        # Fit a weighted average for each budget
        thresholds: List[np.ndarray] = []
        power_array = np.array([get_avg_power(i+1, self._seq_length, self._power_multiplier) for i in range(self._num_levels)])

        for budget in self._budgets:
            distribution = DistributionPrior(power_array, target=budget)
            distribution.make()
            distribution.init()
            
            self._threshold_dict[budget] = distribution.fit()

        self._is_fitted = True

    def predict_sample(self, stop_probs: np.ndarray, budget: int) -> int:
        """
        Predicts the number of levels given the list of hidden states. The states are assumed to be in order.

        Args:
            stop_probs: An [L] array of stop probabilities
            budget: The budget to perform inference under. This controls the employed thresholds.
        Returns:
            The number of levels to execute.
        """
        assert self._is_fitted, 'Model is not fitted'

        # Get thresholds for this budget if needed to infer
        thresholds = self._threshold_dict[budget]
        levels = np.arange(thresholds.shape[0])  # [L]
        chosen_level = self._rand.choice(levels, p=thresholds)
        return chosen_level, get_avg_power(chosen_level + 1, seq_length=self._seq_length, multiplier=self._power_multiplier)


class SkipRNNController(Controller):

    def __init__(self, sample_counts: List[np.ndarray], seq_length: int):
        model_power: List[float] = []
        for counts in sample_counts:
            power = get_weighted_avg_power(counts, seq_length=seq_length)
            model_power.append(power)

        self._model_power = np.array(model_power)

    def fit(self, series: DataSeries):
        pass

    def predict_sample(self, stop_probs: np.ndarray, budget: float) -> int:
        """
        Predicts the number of levels given the list of hidden states. The states are assumed to be in order.

        Args:
            stop_probs: An [L] array of stop probabilities
            budget: The budget to perform inference under. This controls the employed thresholds.
        Returns:
            The number of levels to execute.
        """
        budget_diff = np.abs(self._model_power - budget)
        greater_mask = (self._model_power > budget).astype(float) * BIG_NUMBER

        model_idx = np.argmin(budget_diff + greater_mask)

        return model_idx, self._model_power[model_idx]


class BudgetWrapper:

    def __init__(self, model_predictions: np.ndarray, controller: Controller, max_time: int, seq_length: int, num_classes: int, num_levels: int, budget: float, seed: int = 72):
        self._controller = controller
        self._num_classes = num_classes
        self._seq_length = seq_length
        self._num_levels = num_levels
        self._model_predictions = model_predictions
        self._power_budget = budget
        self._energy_budget = budget * max_time

        # Save variables corresponding to the budget
        self._max_time = max_time
        self._power_results: List[float] = []
        self._energy_margin = 0.05  # Small margin to prevent going over the budget unknowingly

        # Create random state for reproducible results
        self._rand = np.random.RandomState(seed=seed)

    def predict_sample(self, stop_probs: np.ndarray, current_time: int, budget: int, noise: float) -> Tuple[Optional[int], int, float]:
        """
        Predicts the label for the given inputs.

        Args:
            stop_probs: An [L] array of the stop probabilities for each level.
            current_time: The current time index
            budget: The power budget
            noise: The noise on the power reading
        Returns:
            A tuple of two element: (1) A classification for the t-th sample (given by current time)
                (2) The average power consumed to produce this classification
        """
        # Calculate used energy to determine whether to use the model
        used_energy = self.get_consumed_energy()
        should_use_controller = bool(used_energy < self._energy_budget - self._energy_margin)

        # By acting randomly, we incur no energy (no need to collect input samples)
        if not should_use_controller:
            pred = None
            level = 0
            power = 0
        else:
            # If not acting randomly, we use the neural network to perform the classification.
            level, power = self._controller.predict_sample(stop_probs=stop_probs, budget=budget)
            pred = self._model_predictions[current_time, level]

            # Add to power results. If no power is given, we default to using the known
            # power estimates. This gives the controllers a chance to override the power readings.
            if power is None:
                power = get_avg_power(level + 1, seq_length=self._seq_length, multiplier=int(self._seq_length / self._num_levels))

            power = power + noise

        self._power_results.append(power)

        return pred, level, power

    @property
    def power(self) -> List[float]:
        return self._power_results

    def get_consumed_energy(self) -> float:
        return np.sum(self._power_results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-paths', type=str, nargs='+', required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--budgets', type=float, nargs='+', required=True)
    parser.add_argument('--precision', type=int, required=True)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--min-iter', type=int, default=20)
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
                                patience=args.patience,
                                max_iter=args.max_iter,
                                min_iter=args.min_iter)

        # Fit the model on the validation set
        controller.fit(series=DataSeries.VALID)
        controller.save()
