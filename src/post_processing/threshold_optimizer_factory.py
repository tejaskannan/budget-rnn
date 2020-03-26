from .threshold_optimizer import ThresholdOptimizer
from .genetic_optimizer import GeneticOptimizer
from .sim_anneal_optimizer import SimAnnealOptimizer


def get_optimizer(name: str, iterations: int, batch_size: int, level_weight: float, **kwargs) -> ThresholdOptimizer:
    name = name.lower()

    if name == 'genetic':
        return GeneticOptimizer(population_size=kwargs['population_size'],
                                mutation_rate=kwargs['mutation_rate'],
                                batch_size=batch_size,
                                level_weight=level_weight,
                                crossover_rate=kwargs['crossover_rate'],
                                crossover_type=kwargs['crossover_type'],
                                mutation_type=kwargs['mutation_type'],
                                steady_state_count=kwargs['steady_state_count'],
                                iterations=iterations)
    elif name in ('simulated_anneal', 'simulated-anneal', 'sim_anneal', 'sim-anneal'):
        return SimAnnealOptimizer(instances=kwargs['instances'],
                                  epsilon=kwargs['epsilon'],
                                  anneal=kwargs['anneal'],
                                  num_candidates=kwargs['num_candidates'],
                                  move_norm=kwargs['move_norm'],
                                  batch_size=batch_size,
                                  iterations=iterations,
                                  level_weight=level_weight)
    else:
        raise ValueError(f'Unknown optimizer: {name}')
