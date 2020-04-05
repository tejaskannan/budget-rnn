import numpy as np

from typing import List, Tuple, Any

from utils.np_utils import clip_by_norm
from utils.constants import ONE_HALF
from .genetic_optimizer import GeneticOptimizer, CrossoverType, MutationType, LOWER_BOUND, UPPER_BOUND, MAX_NORM


class MulticlassGeneticOptimizer(GeneticOptimizer):
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
                 mode: str,
                 num_classes: int):
        super().__init__(population_size, crossover_rate, mutation_rate, crossover_type, mutation_type, steady_state_count, should_sort, batch_size, iterations, level_weight, mode)
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_classes
 
    def init(self, num_features: int) -> List[Any]:
        population: List[Any] = []
        for _ in range(self.population_size - 1):
            init = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(num_features, self.num_classes))
            population.append(init)

        # Always initialize a threshold with an all 0.5 distribution
        threshold = np.full(shape=(num_features, self.num_classes), fill_value=0.5)
        population.append(threshold)

        return population

    def selection(self, population: List[Any], fitness: List[float]) -> List[Any]:
        population_size = len(population)

        # Select next population using roulette wheel selection
        next_indices = self.sample(fitness, count=population_size - self.steady_state_count)
        next_population = [np.copy(population[i]) for i in next_indices]

        # Perform crossover
        crossover_population: List[np.ndarray] = []
        for i in range(0, len(next_population), 2):
            r = np.random.uniform(low=0.0, high=1.0)

            first_offspring = next_population[i]
            second_offspring = next_population[i+1]

            if r < self.crossover_rate:
                first_offspring, second_offspring = self.crossover(first_offspring, second_offspring)

            crossover_population.append(first_offspring)
            crossover_population.append(second_offspring)

        # Add in the 'best' individuals unchanged. This portion maintains a 'steady state'
        top_indices = np.argsort(fitness)[-self.steady_state_count:]
        crossover_population.extend((np.copy(population[i]) for i in top_indices))

        # Pad population if needed
        for i in range(0, len(population) - len(crossover_population)):
            rand_individual = np.random.randint(low=0, high=len(population))
            crossover_population.append(np.copy(population[rand_individual]))

        return crossover_population

    def mutation(self, population: List[Any]) -> List[Any]:
        mutated_population: List[np.ndarray] = []
        for individual in population:
            if self.mutation_type == MutationType.ELEMENT:
                r = np.random.uniform(low=0.0, high=1.0, size=individual.shape)
                random_elements = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=individual.shape)

                mutated_element = np.where(r < self.mutation_rate, random_elements, individual)
                individual = mutated_element
            elif self.mutation_type == MutationType.NORM:
                r = np.random.uniform(low=0.0, high=1.0)
                if r < self.mutation_rate:
                    random_move = clip_by_norm(np.random.normal(loc=0.0, scale=1.0, size=individual.shape), MAX_NORM, axis=-1)
                    individual = np.clip(individual + random_move, a_min=0.0, a_max=1.0)
            else:
                raise ValueError(f'Unknown mutation type: {self.mutation_type}')
                
            mutated_population.append(individual)

        return mutated_population

    def crossover(self, first_parent: np.ndarray, second_parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert first_parent.shape == second_parent.shape, 'Misaligned individuals'
        assert len(first_parent.shape) == 2, 'Individuals must have 2 dimensions'

        num_levels, num_classes = first_parent.shape

        if self.crossover_type == CrossoverType.ONE_POINT:
            crossover_point = np.random.randint(low=0, high=num_levels - 1)

            temp = np.copy(first_parent[crossover_point:, :])
            first_parent[crossover_point:, :] = second_parent[crossover_point:, :]
            second_parent[crossover_point:, :] = temp
        elif self.crossover_type == CrossoverType.TWO_POINT:
            lower_point = np.random.randint(low=0, high=num_levels - 2)
            upper_point = np.random.randint(low=lower_point + 1, high=num_levels) + 1

            temp = np.copy(first_parent[lower_point:upper_point, :])
            first_parent[lower_point:upper_point, :] = second_parent[lower_point:upper_point, :]
            second_parent[lower_point:upper_point, :] = temp
        elif self.crossover_type == CrossoverType.DIFFERENTIAL:
            weights = np.random.uniform(low=0.0, high=1.0, size=(2, ))

            next_first_parent = first_parent + weights[0] * (second_parent - first_parent)
            next_second_parent = second_parent + weights[1] * (first_parent - second_parent)

            first_parent = np.clip(next_first_parent, a_min=0.0, a_max=1.0)
            second_parent = np.clip(next_second_parent, a_min=0.0, a_max=1.0)
        elif self.crossover_type == CrossoverType.WEIGHTED_AVG:
            weights = np.random.uniform(low=0.0, high=1.0, size=(2, ))

            next_first_parent = weights[0] * first_parent + (1.0 - weights[0]) * second_parent
            next_second_parent = weights[1] * first_parent + (1.0 - weights[1]) * second_parent

            first_parent = np.clip(next_first_parent, a_min=0.0, a_max=1.0)
            second_parent = np.clip(next_second_parent, a_min=0.0, a_max=1.0)
        elif self.crossover_type == CrossoverType.UNIFORM:
            probs = np.random.uniform(low=0.0, high=1.0, size=(num_levels, )) 
            threshold_probs = np.expand_dims(np.less(probs, ONE_HALF), axis=-1)  # [L, 1]
            crossover_locations = np.tile(threshold_probs, reps=(1, num_classes))  # [L, K]

            next_first_parent = np.where(crossover_locations, first_parent, second_parent)
            next_second_parent = np.where(crossover_locations, second_parent, first_parent)

            first_parent, second_parent = next_first_parent, next_second_parent
        else:
            raise ValueError(f'Unknown crossover type: {self.crossover_type}')

        return first_parent, second_parent
