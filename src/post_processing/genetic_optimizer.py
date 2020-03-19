import numpy as np
from typing import List

from .threshold_optimizer import ThresholdOptimizer
from utils.np_utils import softmax, linear_normalize


class GeneticOptimizer(ThresholdOptimizer):

    def __init__(self, population_size: int, crossover_rate: float, mutation_rate: float, batch_size: int, iterations: int):
        super().__init__(iterations, batch_size)

        # Validate parameters
        assert mutation_rate <= 1.0 and mutation_rate >= 0.0, f'Mutation rate must be in [0, 1]'
        assert iterations > 0, f'Must have a positive number of iterations'
        assert crossover_rate <= 1.0 and crossover_rate >= 0.0, f'Crossover rate must be in [0, 1]'

        self._crossover_rate = crossover_rate
        self._population_size = population_size
        self._mutation_rate = mutation_rate

    @property
    def population_size(self):
        return self._population_size

    @property
    def crossover_rate(self):
        return self._crossover_rate

    @property
    def mutation_rate(self):
        return self._mutation_rate

    def init(self, num_features: int) -> List[np.ndarray]:
        population = []
        for _ in range(self.population_size - 1):
            init = np.random.uniform(low=0.0, high=1.0, size=(num_features, ))
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
