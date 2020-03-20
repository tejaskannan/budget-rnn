import numpy as np
from typing import List, Tuple
from enum import Enum, auto

from .threshold_optimizer import ThresholdOptimizer
from utils.np_utils import softmax, linear_normalize


class CrossoverType(Enum):
    ONE_POINT = auto()
    TWO_POINT = auto()
    DIFFERENTIAL = auto()
    WEIGHTED_AVG = auto()
    MASKED_WEIGHTED_AVG = auto()


class GeneticOptimizer(ThresholdOptimizer):

    def __init__(self, population_size: int, crossover_rate: float, mutation_rate: float, crossover_type: str, steady_state_count: int, batch_size: int, iterations: int):
        super().__init__(iterations, batch_size)

        # Validate parameters
        assert mutation_rate <= 1.0 and mutation_rate >= 0.0, f'Mutation rate must be in [0, 1]'
        assert iterations > 0, f'Must have a positive number of iterations'
        assert crossover_rate <= 1.0 and crossover_rate >= 0.0, f'Crossover rate must be in [0, 1]'

        self._crossover_rate = crossover_rate
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._crossover_type = CrossoverType[crossover_type.upper()]
        self._steady_state_count = steady_state_count

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def crossover_rate(self) -> float:
        return self._crossover_rate

    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate

    @property
    def crossover_type(self) -> CrossoverType:
        return self._crossover_type

    @property
    def steady_state_count(self) -> int:
        return self._steady_state_count

    def init(self, num_features: int) -> List[np.ndarray]:
        population = []
        for _ in range(self.population_size - 1):
            init = np.random.uniform(low=0.1, high=0.9, size=(num_features, ))
            population.append(np.sort(init))

        # Always initialize with an all-0.5 distribution
        population.append(np.full(shape=(num_features,), fill_value=0.5))

        return population

    def update(self, state: List[np.ndarray], fitness: List[float], probabilities: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
        state = self._selection(state, fitness)
        return self._mutation(state)

    def _selection(self, population: List[np.ndarray], fitness: List[float]) -> List[np.ndarray]:
        """
        Stochastic universal sampling using a linear ranking fitness technique
        """
        population_size = len(population)
 
        # Select next population using roulette wheel selection
        next_indices = self._sample(fitness, count=population_size - self.steady_state_count)
        next_population = [np.copy(population[i]) for i in next_indices]

        # print('Before crossover: {0}'.format(next_population))

        # Perform crossover
        crossover_population: List[np.ndarray] = []
        for i in range(0, len(next_population), 2):
            first_parent = next_population[i]
            second_parent = next_population[i+1]

            r = np.random.uniform(low=0.0, high=1.0)
            if r < self.crossover_rate:
                first_parent, second_parent = self._crossover(first_parent, second_parent)

            crossover_population.append(np.copy(first_parent))
            crossover_population.append(np.copy(second_parent))

        # Add in the 'best' individuals unchanged
        top_indices = np.argsort(fitness)[-self.steady_state_count:]
        crossover_population.extend((np.copy(population[i]) for i in top_indices))

        # Pad population if needed
        for i in range(0, len(population) - len(crossover_population)):
            rand_individual = np.random.randint(low=0, high=len(population))
            crossover_population.append(np.copy(population[rand_individual]))

        # print('After crossover: {0}'.format(crossover_population))

        return crossover_population

    def _sample(self, fitness: List[float], count: int) -> List[int]:
        normalized_fitness = softmax(fitness)
        indices = list(range(len(fitness)))

        selected: List[int] = []
        for i in range(0, count, 2):
            rand_indices = np.random.choice(indices, 2, replace=False, p=normalized_fitness)
            selected.extend(rand_indices)

        return selected

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

    def _crossover(self, first_parent: np.ndarray, second_parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.crossover_type == CrossoverType.ONE_POINT:
            point = np.random.randint(low=0, high=len(first_parent) - 1)
            
            temp = np.copy(first_parent[point:])
            first_parent[point:] = second_parent[point:]
            return first_parent, second_parent
        elif self.crossover_type == CrossoverType.TWO_POINT:
            lower = np.random.randint(low=0, high=len(first_parent) - 2)
            upper = np.random.randint(low=lower + 1, high=len(first_parent) - 1)

            temp = np.copy(first_parent[lower:upper])
            first_parent[lower:upper] = second_parent[lower:upper]
            second_parent[lower:upper] = temp
        elif self.crossover_type == CrossoverType.DIFFERENTIAL:
            weights = np.random.uniform(low=0.0, high=1.0, size=(2,))

            # Update via a weighted difference
            next_first_parent = first_parent + weights[0] * (second_parent - first_parent)
            next_second_parent = second_parent + weights[1] * (first_parent - second_parent)

            # Clip all values into [0, 1]
            first_parent = np.clip(next_first_parent, a_min=0.0, a_max=1.0)
            second_parent = np.clip(next_second_parent, a_min=0.0, a_max=1.0)
        elif self.crossover_type == CrossoverType.WEIGHTED_AVG:
            weights = np.random.uniform(low=0.0, high=1.0, size=(2, first_parent.shape[0]))

            # Update via a weighed average
            next_first_parent = weights[0] * first_parent + (1.0 - weights[0]) * second_parent
            next_second_parent = weights[1] * first_parent + (1.0 - weights[1]) * second_parent

            # Clip all values into [0, 1]
            first_parent = np.clip(next_first_parent, a_min=0.0, a_max=1.0)
            second_parent = np.clip(next_second_parent, a_min=0.0, a_max=1.0)
        elif self.crossover_type == CrossoverType.MASKED_WEIGHTED_AVG:
            weights = np.random.uniform(low=0.0, high=1.0, size=(2, first_parent.shape[0]))

            num_nonzero = np.random.randint(low=0, high=first_parent.shape[0], size=(2, ))
            first_nonzero_indices = np.random.randint(low=0, high=first_parent.shape[0], size=(num_nonzero[0], ))
            second_nonzero_indices = np.random.randint(low=0, high=first_parent.shape[0], size=(num_nonzero[1], ))

            mask = np.zeros_like(weights)
            mask[0, first_nonzero_indices] = 1
            mask[1, second_nonzero_indices] = 1

            # Apply random mask
            weights = weights * mask

            # Update via a weighed average
            next_first_parent = weights[0] * second_parent + (1.0 - weights[0]) * first_parent
            next_second_parent = weights[1] * first_parent + (1.0 - weights[1]) * second_parent

            # Clip all values into [0, 1]
            first_parent = np.clip(next_first_parent, a_min=0.0, a_max=1.0)
            second_parent = np.clip(next_second_parent, a_min=0.0, a_max=1.0)
 
        return np.sort(first_parent), np.sort(second_parent)
