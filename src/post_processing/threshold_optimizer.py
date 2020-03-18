import numpy as np
from collections import namedtuple
from typing import List, Tuple, Dict, Iterable, Any

from dataset.dataset import DataSeries
from dataset.rnn_sample_dataset import RNNSampleDataset
from models.rnn_model import RNNModel
from utils.rnn_utils import get_logits_name
from utils.misc import softmax, sigmoid
from utils.constants import SMALL_NUMBER
from utils.np_utils import thresholded_predictions, f1_score


OptimizerOutput = namedtuple('OptimizerOutput', ['thresholds', 'score'])

LEVEL_WEIGHT = 0.1

class ThresholdOptimizer:
    """
    Optimizes probability thresholds using a genetic algorithm.
    """

    def __init__(self, population_size: int, mutation_rate: float, batch_size: int, selection_count: int, iterations: int):
        assert population_size > selection_count, f'Must have a larger population than selection count.'
        assert mutation_rate <= 1.0 and mutation_rate >= 0.0, f'Mutation rate must be in [0, 1]'
        assert iterations > 0, f'Must have a positive number of iterations'

        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._batch_size = batch_size
        self._selection_count = selection_count
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
    def selection_count(self) -> int:
        return self._selection_count

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

                selected_pop, selected_fitness = self._selection(population, fitness)
                population = self._crossover(selected_pop, selected_fitness)
                population = self._mutation(population)

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
            init = np.sort(np.random.uniform(low=0.0, high=1.0, size=(num_features, )))
            population.append(init)

        # Always initialize with an all-0.5 distribution
        population.append(np.full(shape=(num_features,), fill_value=0.5))

        return population

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
        Roulette wheel fitness selection
        """
        normalized_fitness = softmax(fitness)

        selected_pop, selected_fitness = [], []
        for i in range(self.selection_count):
            r = np.random.uniform(low=0.0, high=1.0)

            score_sum = 0.0
            index = 0
            for fit_score in normalized_fitness:
                score_sum += fit_score
                if r < score_sum:
                    break
                index += 1    

            selected_pop.append(population[index])
            selected_fitness.append(population[index])


        return selected_pop, selected_fitness

    def _crossover(self, population: List[np.ndarray], fitness: List[float]) -> List[np.ndarray]:
        normalized_fitness = np.array(fitness) / np.sum(fitness)
        num_features = len(population[0])

        iterations = int((self.population_size - self.selection_count) / 2)

        for _ in range(iterations):
            indices = np.random.randint(low=0, high=len(population), size=(2,))

            # Choose random parents
            first_parent = population[indices[0]]
            second_parent = population[indices[1]]

            # Perform crossover
            crossover_point = np.random.randint(low=0, high=num_features)
            temp = first_parent[crossover_point:]
            first_parent[crossover_point:] = second_parent[crossover_point:]
            second_parent[crossover_point:] = temp

            # Sort results
            first_parent = np.sort(first_parent)
            second_parent = np.sort(second_parent)

            population.append(first_parent)
            population.append(second_parent)

        return population

    def _mutation(self, population: List[np.ndarray]) -> List[np.ndarray]:
        for i in range(len(population)):

            individual = np.array(population[i], copy=True)

            for j in range(len(individual)):
                r = np.random.uniform(low=0.0, high=1.0)
                if r < self.mutation_rate:
                    individual[j] = np.random.uniform(low=0.0, high=1.0)

            population[i] = np.sort(individual)

        return population
