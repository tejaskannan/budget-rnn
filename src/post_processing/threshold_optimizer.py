import numpy as np
from collections import namedtuple
from typing import List, Tuple, Dict, Iterable, Any

from dataset.dataset import DataSeries
from dataset.rnn_sample_dataset import RNNSampleDataset
from models.rnn_model import RNNModel
from utils.rnn_utils import get_logits_name
from utils.constants import SMALL_NUMBER
from utils.np_utils import thresholded_predictions, f1_score, softmax, sigmoid, linear_normalize


OptimizerOutput = namedtuple('OptimizerOutput', ['thresholds', 'score'])

LEVEL_WEIGHT = 0.1


class ThresholdOptimizer:
    """
    Optimizes probability thresholds using a genetic algorithm.
    """

    def __init__(self, population_size: int, crossover_rate: float, mutation_rate: float, batch_size: int, iterations: int):
        assert mutation_rate <= 1.0 and mutation_rate >= 0.0, f'Mutation rate must be in [0, 1]'
        assert iterations > 0, f'Must have a positive number of iterations'
        assert crossover_rate <= 1.0 and crossover_rate >= 0.0, f'Crossover rate must be in [0, 1]'

        self._crossover_rate = crossover_rate
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._batch_size = batch_size
        self._iterations = iterations

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def crossover_rate(self) -> float:
        return self._crossover_rate

    @property
    def iterations(self) -> int:
        return self._iterations

    def optimize(self, model: RNNModel, dataset: RNNSampleDataset) -> OptimizerOutput:
        """
        Runs the genetic algorithm optimization.
        """
        # Create data iterator to compute fitness
        data_generator = self._get_data_generator(dataset, model.metadata)

        # Initialize the population
        population = self._init_population(model.num_outputs)

        # Set logit operations
        logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]

        best_score = 0.0
        best_thresholds = [0.5 for _ in range(model.num_outputs)]

        # Compute optimization steps per batch
        batch = next(data_generator)
        for batch_num in range(self.iterations):
            feed_dict = model.batch_to_feed_dict(batch, is_train=False)
            logits = model.execute(feed_dict, logit_ops)

            # Concatenate logits into a 2D array (logit_ops is already ordered by level)
            logits_concat = np.concatenate([logits[op] for op in logit_ops], axis=-1)
            probabilities = sigmoid(logits_concat)
            labels = np.squeeze(np.vstack(batch['output']), axis=-1)
 
            fitness = self._compute_fitness(population, probabilities, labels)

            # Avoid cases in which all labels are zero (both true positive and false negative rates will be zero)
            if any([abs(x) > SMALL_NUMBER for x in labels]):
                best_individual = np.argmax(fitness)
                if fitness[best_individual] > best_score:
                    best_score = fitness[best_individual]
                    best_thresholds = population[best_individual]

                population = self._selection(population, fitness)
                population = self._mutation(population)

            if self._has_converged(population):
                print(f'Converged in {batch_num + 1} iterations. Best score so far: {best_score:.3f}.', end='\r')
                break

            try:
                batch = next(data_generator)
            except StopIteration:
                data_generator = self._get_data_generator(dataset, model.metadata)
                batch = next(data_generator)

            print(f'Completed {batch_num + 1} / {self.iterations} iterations. Best score so far: {best_score:.3f}.', end='\r')

        print()

        return OptimizerOutput(score=best_score, thresholds=best_thresholds)

    def _get_data_generator(self, dataset: RNNSampleDataset, metadata: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        return dataset.minibatch_generator(DataSeries.VALID,
                                           batch_size=self.batch_size,
                                           metadata=metadata,
                                           should_shuffle=True,
                                           drop_incomplete_batches=False)

    def _init_population(self, num_features: int) -> List[np.ndarray]:
        population = []
        for _ in range(self.population_size - 1):
            init = np.random.uniform(low=0.0, high=1.0, size=(num_features, ))
            population.append(np.sort(init))

        # Always initialize with an all-0.5 distribution
        population.append(np.full(shape=(num_features,), fill_value=0.5))

        return population

    def _has_converged(self, population: List[np.ndarray]) -> bool:
        return np.isclose(np.array(population), population[0]).all()

    def _compute_fitness(self, population: List[np.ndarray], probabilities: np.ndarray, labels: np.ndarray) -> List[float]:
        fitnesses: List[float] = []
        
        for individual in population:
            output = thresholded_predictions(probabilities, individual)
            predictions = output.predictions
            levels = output.indices

            num_levels = probabilities.shape[1]
            level_penalty = LEVEL_WEIGHT * np.average((num_levels - levels) / num_levels)
            fitness = f1_score(predictions, labels) + level_penalty

            fitnesses.append(fitness)

        return fitnesses

    def _selection(self, population: List[np.ndarray], fitness: List[float]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Stochastic universal sampling using a linear ranking fitness technique
        """
        population_size = len(population)
 
        # Select next population using roulette wheel selection
        next_indices = self._sample(fitness)
        next_population = [np.copy(population[i]) for i in next_indices]

        # Perform crossover
        crossover_population: List[np.ndarray] = []
        for i in range(0, len(next_population) - 1, 2):
            first_parent = next_population[i]
            second_parent = next_population[i+1]
            
            r = np.random.uniform(low=0.0, high=1.0)
            if r < self.crossover_rate:
                crossover_point = np.random.randint(low=0, high=len(first_parent) - 1)

                temp = np.copy(first_parent[crossover_point:])
                first_parent[crossover_point:] = second_parent[crossover_point:]
                second_parent[crossover_point:] = temp

            crossover_population.append(np.copy(first_parent))
            crossover_population.append(np.copy(second_parent))

        # Pad population if needed
        for i in range(0, len(population) - len(crossover_population)):
            rand_individual = np.random.randint(low=0, high=len(population))
            crossover_population.append(np.copy(population[rand_individual]))

        return crossover_population

    def _sample(self, fitness: List[float]) -> List[int]:
        normalized_fitness = softmax(fitness)
        indices = np.random.choice(a=list(range(len(fitness))),
                                   size=len(fitness),
                                   replace=True,
                                   p=normalized_fitness)
        return indices

    def _mutation(self, population: List[np.ndarray]) -> List[np.ndarray]:
        for i in range(len(population)):

            individual = np.copy(population[i])

            for j in range(len(individual)):
                r = np.random.uniform(low=0.0, high=1.0)
                if r < self.mutation_rate:
                    lower = individual[j-1] if j > 0 else 0.0
                    upper = individual[j+1] if j < len(individual) - 1 else 1.0
                    individual[j] = np.random.uniform(low=lower, high=upper)

            population[i] = np.sort(individual)

        return population
