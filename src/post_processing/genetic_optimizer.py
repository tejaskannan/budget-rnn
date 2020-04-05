import numpy as np
from typing import List, Tuple
from enum import Enum, auto
from copy import deepcopy

from .randomized_threshold_optimizer import RandomizedThresholdOptimizer
from utils.np_utils import softmax, linear_normalize, clip_by_norm
from utils.constants import ONE_HALF
from utils.threshold_utils import TwoSidedThreshold, order_threshold_lists


class CrossoverType(Enum):
    ONE_POINT = auto()
    TWO_POINT = auto()
    DIFFERENTIAL = auto()
    WEIGHTED_AVG = auto()
    MASKED_WEIGHTED_AVG = auto()
    UNIFORM = auto()


class MutationType(Enum):
    ELEMENT = auto()
    NORM = auto()


UPPER_BOUND = 0.0
LOWER_BOUND = 1.0
MAX_NORM = 0.05


class GeneticOptimizer(RandomizedThresholdOptimizer):

    def __init__(self,
                 population_size: int,
                 crossover_rate: float,
                 mutation_rate: float,
                 crossover_type: str,
                 mutation_type: str,
                 steady_state_count: int,
                 should_sort: bool,
                 batch_size: int,
                 iterations: int,
                 level_weight: float,
                 mode: str):
        super().__init__(iterations, batch_size, level_weight, mode)

        # Validate parameters
        assert mutation_rate <= 1.0 and mutation_rate >= 0.0, f'Mutation rate must be in [0, 1]'
        assert iterations > 0, f'Must have a positive number of iterations'
        assert crossover_rate <= 1.0 and crossover_rate >= 0.0, f'Crossover rate must be in [0, 1]'

        self._crossover_rate = crossover_rate
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._crossover_type = CrossoverType[crossover_type.upper()]
        self._mutation_type = MutationType[mutation_type.upper()]
        self._steady_state_count = steady_state_count
        self._should_sort = should_sort

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
    def mutation_type(self) -> MutationType:
        return self._mutation_type

    @property
    def steady_state_count(self) -> int:
        return self._steady_state_count

    @property
    def should_sort(self) -> bool:
        return self._should_sort

    def init(self, num_features: int) -> List[List[TwoSidedThreshold]]:
        population = []
        for _ in range(self.population_size - 1):
            init = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(num_features, 2))

            thresholds = [TwoSidedThreshold(lower=np.min(x), upper=np.max(x)) for x in init]
            population.append(thresholds)

        # Always initialize with an all-0.5 distribution
        threshold = [TwoSidedThreshold(lower=0.5, upper=1.0) for _ in range(num_features)]
        population.append(threshold)

        return population

    def update(self, state: List[List[TwoSidedThreshold]], fitness: List[float], probabilities: np.ndarray, labels: np.ndarray) -> List[List[TwoSidedThreshold]]:
        state = self.selection(state, fitness)
        return self.mutation(state)

    def selection(self, population: List[List[TwoSidedThreshold]], fitness: List[float]) -> List[List[TwoSidedThreshold]]:
        """
        Stochastic universal sampling using a linear ranking fitness technique
        """
        population_size = len(population)
 
        # Select next population using roulette wheel selection
        next_indices = self.sample(fitness, count=population_size - self.steady_state_count)
        next_population = [deepcopy(population[i]) for i in next_indices]

        # Perform crossover
        crossover_population: List[TwoSidedThreshold] = []
        for i in range(0, len(next_population), 2):

            # Crossover on thresholds
            first_parent = next_population[i]
            second_parent = next_population[i+1]

            # Get lower and upper thresholds
            first_lower, first_upper = [t.lower for t in first_parent], [t.upper for t in first_parent]
            second_lower, second_upper = [t.lower for t in second_parent], [t.upper for t in second_parent]

            # Crossover on lower
            r = np.random.uniform(low=0.0, high=1.0)
            if r < self.crossover_rate:
                first_lower, second_lower = self.crossover(first_lower, second_lower)

            # Crossover on upper
            r = np.random.uniform(low=0.0, high=1.0)
            if r < self.crossover_rate:
                first_upper, second_upper = self.crossover(first_upper, second_upper)

            # Create new individuals
            first_offspring = order_threshold_lists(lower=first_lower, upper=first_upper, should_sort=self.should_sort)
            second_offspring = order_threshold_lists(lower=second_lower, upper=second_upper, should_sort=self.should_sort)

            crossover_population.append(first_offspring)
            crossover_population.append(second_offspring)

        # Add in the 'best' individuals unchanged
        top_indices = np.argsort(fitness)[-self.steady_state_count:]
        crossover_population.extend((deepcopy(population[i]) for i in top_indices))

        # Pad population if needed
        for i in range(0, len(population) - len(crossover_population)):
            rand_individual = np.random.randint(low=0, high=len(population))
            crossover_population.append(deepcopy(population[rand_individual]))

        return crossover_population

    def sample(self, fitness: List[float], count: int) -> List[int]:
        normalized_fitness = softmax(fitness)
        indices = list(range(len(fitness)))

        selected: List[int] = []
        for i in range(0, count, 2):
            rand_indices = np.random.choice(indices, 2, replace=False, p=normalized_fitness)
            selected.extend(rand_indices)

        return selected

    def mutation(self, population: List[List[TwoSidedThreshold]]) -> List[List[TwoSidedThreshold]]:
        for i in range(len(population)):
            # Get individual thresholds
            lower = [t.lower for t in population[i]]
            upper = [t.upper for t in population[i]]

            # Apply desired mutation operation
            if self.mutation_rate == MutationType.ELEMENT:
                r = np.random.uniform(low=0.0, high=1.0, size=(len(individual), ))
                random_elements = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(len(lower), ))
                lower = np.where(r < self.mutation_rate, random_elements, lower)

                r = np.random.uniform(low=0.0, high=1.0, size=(len(individual), ))
                random_elements = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(len(upper), ))
                upper = np.where(r < self.mutation_rate, random_elements, upper)
            elif self.mutation_rate == MutationType.NORM:
                r = np.random.uniform(low=0.0, high=1.0)
                if r < self.mutation_rate:
                    random_move = clip_by_norm(np.random.normal(loc=0.0, scale=1.0, size=(len(lower), )), MAX_NORM)
                    lower = np.clip(lower + random_move, a_min=0.0, a_max=1.0)

                r = np.random.uniform(low=0.0, high=1.0)
                if r < self.mutation_rate:
                    random_move = clip_by_norm(np.random.normal(loc=0.0, scale=1.0, size=(len(uppper), )), MAX_NORM)
                    upper = np.clip(upper + random_move, a_min=0.0, a_max=1.0)

            # Re-order the mutated individual
            mutated_individual = order_threshold_lists(lower=lower, upper=upper, should_sort=self.should_sort)

            # Include mutated individual in the population
            population[i] = mutated_individual

        return population

    def crossover(self, first_parent: np.ndarray, second_parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        num_features = len(first_parent)

        if self.crossover_type == CrossoverType.ONE_POINT:
            point = np.random.randint(low=0, high=num_features - 1)
            
            temp = np.copy(first_parent[point:])
            first_parent[point:] = second_parent[point:]
            return first_parent, second_parent
        elif self.crossover_type == CrossoverType.TWO_POINT:
            lower = np.random.randint(low=0, high=num_features - 2)
            upper = np.random.randint(low=lower + 1, high=num_features - 1)

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
            weights = np.random.uniform(low=0.0, high=1.0, size=(2, num_features))

            # Update via a weighed average
            next_first_parent = weights[0] * first_parent + (1.0 - weights[0]) * second_parent
            next_second_parent = weights[1] * first_parent + (1.0 - weights[1]) * second_parent

            # Clip all values into [0, 1]
            first_parent = np.clip(next_first_parent, a_min=0.0, a_max=1.0)
            second_parent = np.clip(next_second_parent, a_min=0.0, a_max=1.0)
        elif self.crossover_type == CrossoverType.MASKED_WEIGHTED_AVG:
            weights = np.random.uniform(low=0.0, high=1.0, size=(2, num_features))

            num_nonzero = np.random.randint(low=0, high=first_parent.shape[0], size=(2, ))
            first_nonzero_indices = np.random.randint(low=0, high=num_features, size=(num_nonzero[0], ))
            second_nonzero_indices = np.random.randint(low=0, high=num_features, size=(num_nonzero[1], ))

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
        elif self.crossover_type == CrossoverType.UNIFORM:
            probs = np.random.uniform(low=0.0, high=1.0, size=(num_features, ))

            next_first_parent = np.where(probs < ONE_HALF, first_parent, second_parent)
            next_second_parent = np.where(probs < ONE_HALF, second_parent, first_parent)

            first_parent, second_parent = next_first_parent, next_second_parent

        return first_parent, second_parent
