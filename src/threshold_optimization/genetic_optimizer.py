import numpy as np
from enum import Enum, auto
from sklearn.metrics import f1_score
from typing import Any, Dict, Iterable, Tuple, List, Optional

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from utils.rnn_utils import get_logits_name
from utils.testing_utils import ClassificationMetric
from utils.np_utils import round_to_precision, min_max_normalize, clip_by_norm
from utils.constants import BIG_NUMBER, SMALL_NUMBER, OUTPUT


class CrossoverType(Enum):
    ONE_POINT = auto()
    TWO_POINT = auto()
    WEIGHTED_AVG = auto()
    DIFFERENTIAL = auto()
    UNIFORM = auto()


def threshold_predictions(predictions: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the predictions using the early-stopping inference algorithm.

    Args:
        predictions: [B, L, C] array of normalized log probabilities for each sample, level, and class
        thresholds: [L] array of thresholds for each level
    Returns:
        A tuple of (1) A [B] array with the predictions per sample and (2) A [B] array with the number of computed levels.
    """
    # Reshape thresholds to a [1, L] array
    expanded_thresholds = np.expand_dims(thresholds, axis=0)

    # Create mask using the maximum probability
    max_prob = np.max(predictions, axis=-1)  # [B, L]
    diff_mask = (max_prob < expanded_thresholds).astype(np.float32) * BIG_NUMBER  # [B, L]

    indices = np.expand_dims(np.arange(start=0, stop=len(thresholds)), axis=0)  # [1, L]

    # Apply mask to compute the number of computed levels
    masked_indices = indices + diff_mask  # [B, L]
    levels = np.clip(np.min(masked_indices, axis=-1).astype(int), a_min=0, a_max=predictions.shape[1] - 1)  # [B]

    # Use number of levels to get the classification
    predicted_class_per_level = np.argmax(predictions, axis=-1)  # [B, L]
    batch_indices = np.arange(start=0, stop=predictions.shape[0])  # [B]
    predicted_classes = predicted_class_per_level[batch_indices, levels]  # [B]

    return predicted_classes, levels


def rank_sample(fitness: List[float], count: int) -> np.ndarray:
    indices = list(range(len(fitness)))

    sorted_indices = np.argsort(fitness)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(fitness))
    normalized_ranks = ranks / np.sum(ranks)

    selected_indices = np.random.choice(indices, count, replace=False, p=normalized_ranks)
    return selected_indices


class GeneticThresholdOptimizer:

    def __init__(self, params: Dict[str, Any], model: AdaptiveModel):
        self._model = model

        self._precision = params['precision']
        self._batch_size = params['batch_size']
        self._num_iterations = params['iterations']
        self._population_size = params['population_size']
        self._crossover_type = CrossoverType[params['crossover_strategy'].upper()]
        self._crossover_rate = params['crossover_rate']
        self._mutation_rate = params['mutation_rate']
        self._level_penalty = params['level_penalty']
        self._steady_state_frac = params['steady_state_fraction']
        self._mutation_norm = params['mutation_norm']
        self._should_sort_thresholds = params['should_sort_thresholds']

        self._num_levels = self._model.num_outputs

        self._thresholds = None

    def fit(self, dataset: Dataset, series: DataSeries) -> np.ndarray:
        # Initialize the thresholds. Always start with a single sample with all 0.5.
        state = np.random.uniform(low=0.0, high=1.0, size=(self._population_size, self._num_levels))
        state[-1] = np.full(shape=(self._num_levels, ), fill_value=0.5)
        state = round_to_precision(state, precision=self._precision)

        if self._should_sort_thresholds:
            state = np.sort(state, axis=-1)

        # Set logit operations
        logit_ops = [get_logits_name(i) for i in range(self._num_levels)]

        data_generator = dataset.minibatch_generator(series=series,
                                                     batch_size=self._batch_size,
                                                     metadata=self._model.metadata,
                                                     should_shuffle=True)
        best_individual = None
        for batch_num, batch in enumerate(data_generator):
            if batch_num >= self._num_iterations:
                break

            # Compute the predicted log probabilities
            feed_dict = self._model.batch_to_feed_dict(batch, is_train=False)
            logits = self._model.execute(feed_dict, logit_ops)

            # Concatenate logits into a [B, L, C] array (logit_ops is already ordered by level).
            # For reference, L is the number of levels and C is the number of classes
            logits_concat = np.concatenate([np.expand_dims(logits[op], axis=1) for op in logit_ops], axis=1)

            # Normalize logits and round to fixed point representation
            normalized_logits = min_max_normalize(logits_concat, axis=-1)
            normalized_logits = round_to_precision(normalized_logits, precision=self._precision)

            # Compute fitness values and transition the population
            fitnesses = self.evaluate(state, normalized_logits, batch[OUTPUT])
            state = self.selection(state, fitnesses)
            state = self.mutation(state)

            state = round_to_precision(state, precision=self._precision)
            if self._should_sort_thresholds:
                state = np.sort(state, axis=-1)

            best_idx = np.argmax(fitnesses)
            best_individual = np.copy(state[best_idx])
            best_fitness = fitnesses[best_idx]

            print('Completed batch {0}. Best Fitness: {1:.4f}'.format(batch_num + 1, best_fitness), end='\r')

            # Detect if the process has converged
            if np.isclose(state, state[0]).all():
                print('Converged after {0} iterations. Best Fitness: {1:.4f}'.format(batch_num + 1, best_fitness), end='\r')
        print()

        # Set the best thresholds
        self._thresholds = best_individual

        return best_individual

    def score(self, dataset: Dataset, series: DataSeries, flops_per_level: List[float], thresholds: Optional[np.ndarray] = None):
        assert self._thresholds is not None or thresholds is not None, 'Must fit the model or provide thresholds.'

        if thresholds is None:
            thresholds = self._thresholds

        # Set logit operations
        logit_ops = [get_logits_name(i) for i in range(self._num_levels)]

        data_generator = dataset.minibatch_generator(series=series,
                                                     batch_size=self._batch_size,
                                                     metadata=self._model.metadata,
                                                     should_shuffle=False)
 
        predictions: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        levels: List[np.ndarray] = []
        flops: List[np.ndarray] = []

        for batch in data_generator:
            # Compute the predicted log probabilities
            feed_dict = self._model.batch_to_feed_dict(batch, is_train=False)
            logits = self._model.execute(feed_dict, logit_ops)

            # Concatenate logits into a [B, L, C] array (logit_ops is already ordered by level).
            # For reference, L is the number of levels and C is the number of classes
            logits_concat = np.concatenate([np.expand_dims(logits[op], axis=1) for op in logit_ops], axis=1)

            # Normalize logits and round to fixed point representation
            normalized_logits = min_max_normalize(logits_concat, axis=-1)
            normalized_logits = round_to_precision(normalized_logits, precision=self._precision)

            batch_predictions, batch_levels = threshold_predictions(normalized_logits, thresholds=thresholds)

            predictions.append(batch_predictions)
            levels.append(batch_levels)
            labels.append(np.squeeze(batch[OUTPUT]))
            flops.append([flops_per_level[ell] for ell in batch_levels])

        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)
        levels = np.concatenate(levels, axis=0) + 1
        flops = np.concatenate(flops, axis=0)

        return {
            ClassificationMetric.ACCURACY.name: np.average((labels == predictions).astype(float)),
            ClassificationMetric.MACRO_F1_SCORE.name: f1_score(labels, predictions, average='macro'),
            ClassificationMetric.MICRO_F1_SCORE.name: f1_score(labels, predictions, average='micro'),
            ClassificationMetric.LEVEL.name: np.average(levels),
            ClassificationMetric.FLOPS.name: np.average(flops),
            'THRESHOLDS': thresholds.astype(float).tolist()
        }

    def evaluate(self, state: np.ndarray, normalized_logits: np.ndarray, labels: np.ndarray) -> List[float]:
        """
        Computes the fitness for each element in the state.

        Args:
            state: A [P, L] set of thresholds for each member of the population (P) and level (L)
            normalized_logits: A [B, L, C] array of normalized logits for each class and level
            labels: A [B] array of true outputs for the given samples.
        Returns:
            Fitness scores for each sample in the population.
        """
        fitness: List[float] = []
        for element in state:
            fitness.append(self.fitness_function(normalized_logits, labels, element))

        return fitness

    def fitness_function(self, normalized_logits: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> float:
        predictions, levels = threshold_predictions(normalized_logits, thresholds=thresholds)

        accuracy = np.average((predictions == labels).astype(float))
        level_penalty = -1 * self._level_penalty * np.average(levels)

        return accuracy + level_penalty

    def selection(self, state: np.ndarray, fitnesses: List[float]) -> np.ndarray:
        """
        Transitions the population by randomly selecting individuals and performing crossover.
        """
        steady_state_samples = int(self._population_size * self._steady_state_frac)

        # Select individuals for crossover
        selected_indices = rank_sample(fitnesses, self._population_size - steady_state_samples)

        # Perform crossover
        new_population: List[np.ndarray] = []
        for i in range(0, self._population_size - steady_state_samples - 1, 2):
            idx1, idx2 = selected_indices[i], selected_indices[i+1]
            first_parent, second_parent = np.copy(state[idx1]), np.copy(state[idx2])

            r = np.random.uniform(low=0.0, high=1.0)
            if r < self._crossover_rate:
                first_offspring, second_offspring = self.crossover(first_parent, second_parent)
            else:
                first_offspring, second_offspring = first_parent, second_parent

            new_population.append(first_offspring)
            new_population.append(second_offspring)

        # Include the remaining individual when the selection size is odd
        if len(new_population) < self._population_size - steady_state_samples:
            new_population.append(np.copy(state[selected_indices[-1]]))

        top_individual_indices = np.argsort(fitnesses)[::-1][:steady_state_samples]
        top_individuals = [np.copy(state[i]) for i in top_individual_indices]
        new_population.extend(top_individuals)

        return np.array(new_population)

    def crossover(self, first_parent: np.ndarray, second_parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert first_parent.shape == second_parent.shape, 'Misaligned individuals'

        num_levels = first_parent.shape[0]

        if self._crossover_type == CrossoverType.ONE_POINT:
            crossover_point = np.random.randint(low=0, high=num_levels - 1)

            temp = np.copy(first_parent[crossover_point:])
            first_parent[crossover_point:] = second_parent[crossover_point:]
            second_parent[crossover_point:] = temp
        elif self._crossover_type == CrossoverType.TWO_POINT:
            lower_point = np.random.randint(low=0, high=num_levels - 2)
            upper_point = np.random.randint(low=lower_point + 1, high=num_levels) + 1

            temp = np.copy(first_parent[lower_point:upper_point])
            first_parent[lower_point:upper_point] = second_parent[lower_point:upper_point]
            second_parent[lower_point:upper_point] = temp
        elif self._crossover_type == CrossoverType.DIFFERENTIAL:
            weights = np.random.uniform(low=0.0, high=1.0, size=(2, ))

            next_first_parent = first_parent + weights[0] * (second_parent - first_parent)
            next_second_parent = second_parent + weights[1] * (first_parent - second_parent)

            first_parent = np.clip(next_first_parent, a_min=0.0, a_max=1.0)
            second_parent = np.clip(next_second_parent, a_min=0.0, a_max=1.0)
        elif self._crossover_type == CrossoverType.WEIGHTED_AVG:
            weights = np.random.uniform(low=0.0, high=1.0, size=(2, ))

            next_first_parent = weights[0] * first_parent + (1.0 - weights[0]) * second_parent
            next_second_parent = weights[1] * first_parent + (1.0 - weights[1]) * second_parent

            first_parent = np.clip(next_first_parent, a_min=0.0, a_max=1.0)
            second_parent = np.clip(next_second_parent, a_min=0.0, a_max=1.0)
        elif self._crossover_type == CrossoverType.UNIFORM:
            probs = np.random.uniform(low=0.0, high=1.0, size=(num_levels, )) 
            threshold_probs = np.expand_dims(np.less(probs, ONE_HALF), axis=-1)  # [L, 1]
            crossover_locations = np.tile(threshold_probs, reps=(1, num_classes))  # [L, K]

            next_first_parent = np.where(crossover_locations, first_parent, second_parent)
            next_second_parent = np.where(crossover_locations, second_parent, first_parent)

            first_parent, second_parent = next_first_parent, next_second_parent
        else:
            raise ValueError(f'Unknown crossover type: {self.crossover_type}')

        return first_parent, second_parent

    def mutation(self, state: np.ndarray) -> np.ndarray:
        mutations = np.random.uniform(low=0.0, high=1.0, size=state.shape)
        mutations = clip_by_norm(mutations, clip=self._mutation_norm, axis=-1)

        mutation_individuals = np.random.uniform(low=0.0, high=1.0, size=(self._population_size, 1))
        masked_mutations = np.where(mutation_individuals < self._mutation_rate, mutations, np.zeros_like(state))

        return state + masked_mutations
