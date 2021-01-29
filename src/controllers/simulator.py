import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import time
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from scipy import integrate
from typing import Tuple, List, Union, Optional, Dict, Any, Set

from controllers.runtime_system import RuntimeSystem, SystemType
from controllers.controller_utils import execute_adaptive_model, execute_standard_model, concat_model_results, LOG_FILE_FMT
from controllers.controller_utils import save_test_log, execute_skip_rnn_model, ModelResults, execute_phased_rnn_model
from controllers.noise_generators import get_noise_generator, NoiseGenerator
from controllers.power_utils import PowerType, make_power_system
from models.base_model import Model
from models.model_factory import get_model
from models.adaptive_model import AdaptiveModel
from models.standard_model import StandardModel
from dataset.dataset import DataSeries, Dataset
from dataset.dataset_factory import get_dataset
from utils.hyperparameters import HyperParameters
from utils.file_utils import extract_model_name, read_by_file_suffix, save_by_file_suffix, make_dir, iterate_files
from utils.constants import SMALL_NUMBER, METADATA_PATH, HYPERS_PATH, SEQ_LENGTH, NUM_CLASSES
from utils.loading_utils import restore_neural_network


SimulationResult = namedtuple('SimulationResult', ['accuracy', 'power', 'target_budgets', 'energy'])
COLORS = ['#08589e', '#6e016b', '#8c96c6', '#41b6c4', '#fd8d3c', '#238443']


def run_simulation(runtime_systems: List[RuntimeSystem], budget: float, max_time: int, noise_generator: NoiseGenerator) -> Tuple[Dict[str, SimulationResult], List[float]]:

    # Initialize the systems for this budget
    for system in runtime_systems:
        system.init_for_budget(budget=budget, max_time=max_time)

    # Sequentially execute each system
    noise_terms: List[float] = []
    for t in range(max_time):
        power_noise = noise_generator.get_noise(t=t)
        noise_terms.append(power_noise)

        for system in runtime_systems:
            system.step(budget=budget, power_noise=power_noise, t=t)

    # Create the final result
    times = np.arange(max_time) + 1
    result: Dict[str, SimulationResult] = dict()

    for system in runtime_systems:
        energy = system.get_energy()

        system_result = SimulationResult(power=energy / times,
                                         accuracy=system.get_num_correct() / times,
                                         energy=energy,
                                         target_budgets=system.get_target_budgets())
        result[system.name] = system_result

    return result, noise_terms


def moving_avg_power(energy: np.ndarray, window: int) -> np.ndarray:
    moving_avg = energy[window:] - energy[:-window]
    return moving_avg[window - 1:] / window


def plot_and_save(sim_results: Dict[str, SimulationResult],
                  runtime_systems: List[RuntimeSystem],
                  output_folder: str,
                  budget: float,
                  max_time: int,
                  noise_generator: NoiseGenerator,
                  noise_terms: List[float],
                  power_system_type: PowerType,
                  should_plot: bool,
                  save_plots: bool,
                  baseline_to_plot: str):
    # Make the output folder if necessary
    make_dir(output_folder)

    # Log the test results for each adaptive system
    model_names: Set[str] = set()

    system_dict = {system.name: system for system in runtime_systems}
    for system_name in sorted(sim_results.keys()):
        system = system_dict[system_name]
        sim_result = sim_results[system_name]

        # We compute the validation accuracy for this budget for the adaptive models.
        # This allows us to choose which backend model to select at testing time.
        if system.system_type == SystemType.ADAPTIVE:
            valid_accuracy = system.estimate_validation_results(budget=budget,
                                                                max_time=max_time)
        else:
            valid_accuracy = None

        model_names.add(system_name.split()[0])

        log_file_name = LOG_FILE_FMT.format(system.system_type.name.lower(), system.model_name, power_system_type.name.lower())
        log_path = os.path.join(output_folder, log_file_name)
        save_test_log(accuracy=sim_result.accuracy[-1],
                      power=sim_result.power[-1],
                      valid_accuracy=valid_accuracy,
                      budget=budget,
                      key=str(noise_generator),
                      system_name=system.name,
                      output_file=log_path)

        print('{0} Accuracy: {1:.5f}, {0} Power: {2:.5f}'.format(system_name, sim_result.accuracy[-1], sim_result.power[-1]))

    if not should_plot:
        return

    # Filter the simulation results
    systems_to_keep = []
    if baseline_to_plot in ('under_budget', 'all'):
        for name in model_names:
            systems_to_keep.append('{0} FIXED_UNDER_BUDGET'.format(name))

    if baseline_to_plot in ('max_accuracy', 'all'):
        for name in model_names:
            systems_to_keep.append('{0} FIXED_MAX_ACCURACY'.format(name))

    sim_results = {system_name: result for system_name, result in sim_results.items() if system_name in systems_to_keep or 'ADAPTIVE' in system_name or 'RANDOMIZED' in system_name}
    colors = {system_name: COLORS[i] for i, system_name in enumerate(sim_results.keys())}

    # List of times for plotting
    times = np.arange(max_time) + 1

    # Plot the results
    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(16, 12), nrows=3, ncols=1, sharex=True)

        # Plot the energy noise terms
        ax1.plot(times, noise_terms, color='#e34a33')
        ax1.set_title('Per-Step Energy Noise')
        ax1.set_ylabel('Energy (mJ)')

        # Plot the Setpoints of each system
        for system_name, sim_result in sorted(sim_results.items()):
            if 'adaptive' in system_name.lower():
                ax2.plot(times, sim_result.target_budgets * max_time, label=system_name, color=colors[system_name])

        ax2.axhline(budget * max_time, color='k', linewidth=2)

        ax2.legend(fontsize=9)
        ax2.set_title('Budget Setpoint')
        ax2.set_ylabel('Energy (mJ)')

        # Plot the Moving Avg Power of each system
        for system_name, sim_result in sorted(sim_results.items()):
            avg_power = moving_avg_power(sim_result.energy, window=20)
            diff = len(times) - len(avg_power)

            ax3.plot(times[diff:], avg_power, label=system_name, color=colors[system_name])

        ax3.axhline(budget, color='k', linewidth=2)

        ax3.legend(loc='lower center', fontsize=8)
        ax3.set_title('Moving Average Power for Each Policy')
        ax3.set_ylabel('Power (mW)')
        ax3.set_xlabel('Time')

        plt.tight_layout()

        if save_plots:
            output_file = os.path.join(output_folder, 'results_{0}.pdf'.format(budget))
            plt.savefig(output_file)
        else:
            plt.show()


def create_multi_model_systems(folder: str, model_type: str, power_system_type: PowerType) -> List[RuntimeSystem]:
    model_type = model_type.upper()
    assert model_type in ('SKIP_RNN', 'PHASED_RNN'), 'Unknown model type: {0}'.format(model_type)

    runtime_systems: List[RuntimeSystem] = []

    valid_results: List[ModelResults] = []
    test_results: List[ModelResults] = []
    model_paths: List[str] = []
    for model_path in iterate_files(folder, pattern=r'model-{0}-.*model_best\.pkl\.gz'.format(model_type)):
        model, dataset = restore_neural_network(model_path, dataset_folder=dataset_folder)

        if model_type == 'SKIP_RNN':
            valid_result = execute_skip_rnn_model(model, dataset, series=DataSeries.VALID)
            test_result = execute_skip_rnn_model(model, dataset, series=DataSeries.TEST)
        else:
            valid_result = execute_phased_rnn_model(model, dataset, series=DataSeries.VALID)
            test_result = execute_phased_rnn_model(model, dataset, series=DataSeries.TEST)

        valid_results.append(valid_result)
        test_results.append(test_result)

        model_paths.append(model_path)

    # Concatenate the results from each model
    test_results_concat = concat_model_results(test_results)
    valid_results_concat = concat_model_results(valid_results)

    power_system = make_power_system(num_levels=seq_length,
                                     seq_length=seq_length,
                                     model_type=model.model_type,
                                     power_type=power_system_type)

    under_budget = RuntimeSystem(test_results=test_results_concat,
                                 valid_results=valid_results_concat,
                                 system_type=SystemType.FIXED_UNDER_BUDGET,
                                 model_path=model_paths[0],
                                 dataset_folder=dataset_folder,
                                 seq_length=seq_length,
                                 num_levels=len(model_paths),
                                 num_classes=num_classes,
                                 power_system=power_system)
    runtime_systems.append(under_budget)

    max_accuracy = RuntimeSystem(test_results=test_results_concat,
                                 valid_results=valid_results_concat,
                                 system_type=SystemType.FIXED_MAX_ACCURACY,
                                 model_path=model_paths[0],
                                 dataset_folder=dataset_folder,
                                 seq_length=seq_length,
                                 num_levels=len(model_paths),
                                 num_classes=num_classes,
                                 power_system=power_system)
    runtime_systems.append(max_accuracy)

    return runtime_systems


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--adaptive-model-paths', type=str, nargs='+', required=True)
    parser.add_argument('--baseline-model-path', type=str, required=True)
    parser.add_argument('--skip-model-folder', type=str)
    parser.add_argument('--phased-model-folder', type=str)
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--budget-start', type=float, required=True)
    parser.add_argument('--budget-end', type=float, required=True)
    parser.add_argument('--budget-step', type=float, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--noise-params', type=str, nargs='+', required=True)
    parser.add_argument('--power-system-type', type=str, choices=['bluetooth', 'temp'], default='temp')
    parser.add_argument('--skip-plotting', action='store_true')
    parser.add_argument('--save-plots', action='store_true')
    parser.add_argument('--baseline-to-plot', type=str, choices=['all', 'under_budget', 'max_accuracy'], default='all')
    args = parser.parse_args()

    # Validate arguments
    budget_start, budget_end, budget_step = args.budget_start, args.budget_end, args.budget_step
    assert budget_start > 0, 'Must have a positive budget'
    assert budget_end >= budget_start, 'Must have budget_end >= budget_start'
    assert budget_step > 0, 'Must have a positive budget step'

    budgets = np.arange(start=budget_start, stop=budget_end + (budget_step / 2), step=budget_step)

    dataset_folder = args.dataset_folder
    power_system_type = PowerType[args.power_system_type.upper()]

    # Make systems based on adaptive models
    runtime_systems: List[RuntimeSystem] = []
    for adaptive_model_path in args.adaptive_model_paths:
        model, dataset = restore_neural_network(adaptive_model_path, dataset_folder=dataset_folder)

        num_levels = model.num_outputs
        seq_length = model.metadata[SEQ_LENGTH]
        num_classes = model.metadata[NUM_CLASSES]

        valid_results = execute_adaptive_model(model, dataset, series=DataSeries.VALID)
        test_results = execute_adaptive_model(model, dataset, series=DataSeries.TEST)

        adaptive_power_system = make_power_system(num_levels=num_levels,
                                                  seq_length=seq_length,
                                                  power_type=power_system_type,
                                                  model_type=model.model_type)

        adaptive_system = RuntimeSystem(valid_results=valid_results,
                                        test_results=test_results,
                                        system_type=SystemType.ADAPTIVE,
                                        model_path=adaptive_model_path,
                                        dataset_folder=dataset_folder,
                                        seq_length=seq_length,
                                        num_levels=num_levels,
                                        num_classes=num_classes,
                                        power_system=adaptive_power_system)
        runtime_systems.append(adaptive_system)

        adaptive_fixed_under_budget = RuntimeSystem(valid_results=valid_results,
                                                    test_results=test_results,
                                                    system_type=SystemType.FIXED_UNDER_BUDGET,
                                                    model_path=adaptive_model_path,
                                                    dataset_folder=dataset_folder,
                                                    seq_length=seq_length,
                                                    num_levels=num_levels,
                                                    num_classes=num_classes,
                                                    power_system=adaptive_power_system)
        runtime_systems.append(adaptive_fixed_under_budget)

        adaptive_fixed_max_accuracy = RuntimeSystem(valid_results=valid_results,
                                                    test_results=test_results,
                                                    system_type=SystemType.FIXED_MAX_ACCURACY,
                                                    model_path=adaptive_model_path,
                                                    dataset_folder=dataset_folder,
                                                    seq_length=seq_length,
                                                    num_levels=num_levels,
                                                    num_classes=num_classes,
                                                    power_system=adaptive_power_system)
        runtime_systems.append(adaptive_fixed_max_accuracy)

        randomized_system = RuntimeSystem(valid_results=valid_results,
                                          test_results=test_results,
                                          system_type=SystemType.RANDOMIZED,
                                          model_path=adaptive_model_path,
                                          dataset_folder=dataset_folder,
                                          seq_length=seq_length,
                                          num_levels=num_levels,
                                          num_classes=num_classes,
                                          power_system=adaptive_power_system)
        runtime_systems.append(randomized_system)

    # Make the baseline systems
    baseline_model_path = args.baseline_model_path
    model, dataset = restore_neural_network(baseline_model_path, dataset_folder=dataset_folder)

    valid_results = execute_standard_model(model, dataset, series=DataSeries.VALID)
    test_results = execute_standard_model(model, dataset, series=DataSeries.TEST)

    seq_length = model.metadata[SEQ_LENGTH]
    num_classes = model.metadata[NUM_CLASSES]

    standard_power_system = make_power_system(num_levels=seq_length,
                                              seq_length=seq_length,
                                              power_type=power_system_type,
                                              model_type=model.model_type)

    greedy_system = RuntimeSystem(test_results=test_results,
                                  valid_results=valid_results,
                                  system_type=SystemType.GREEDY,
                                  model_path=baseline_model_path,
                                  dataset_folder=dataset_folder,
                                  seq_length=seq_length,
                                  num_levels=seq_length,
                                  num_classes=num_classes,
                                  power_system=standard_power_system)
    runtime_systems.append(greedy_system)

    fixed_under_budget = RuntimeSystem(test_results=test_results,
                                       valid_results=valid_results,
                                       system_type=SystemType.FIXED_UNDER_BUDGET,
                                       model_path=baseline_model_path,
                                       dataset_folder=dataset_folder,
                                       seq_length=seq_length,
                                       num_levels=seq_length,
                                       num_classes=num_classes,
                                       power_system=standard_power_system)
    runtime_systems.append(fixed_under_budget)

    fixed_max_accuracy = RuntimeSystem(test_results=test_results,
                                       valid_results=valid_results,
                                       system_type=SystemType.FIXED_MAX_ACCURACY,
                                       model_path=baseline_model_path,
                                       dataset_folder=dataset_folder,
                                       seq_length=seq_length,
                                       num_levels=seq_length,
                                       num_classes=num_classes,
                                       power_system=standard_power_system)
    runtime_systems.append(fixed_max_accuracy)

    # Add the Skip RNN models if provided
    skip_rnn_folder = args.skip_model_folder
    if skip_rnn_folder is not None:
        skip_rnn_systems = create_multi_model_systems(folder=skip_rnn_folder,
                                                      model_type='SKIP_RNN',
                                                      power_system_type=power_system_type)
        runtime_systems.extend(skip_rnn_systems)

    # Add the Phased RNN models if provided
    phased_rnn_folder = args.phased_model_folder
    if phased_rnn_folder is not None:
        phased_rnn_systems = create_multi_model_systems(folder=phased_rnn_folder,
                                                        model_type='PHASED_RNN',
                                                        power_system_type=power_system_type)
        runtime_systems.extend(phased_rnn_systems)

    # Max time equals the number of test samples
    max_time = dataset.dataset[DataSeries.TEST].length

    for noise_params_path in args.noise_params:
        noise_params = read_by_file_suffix(noise_params_path)

        # Create the noise generator for the given parameters
        for noise_generator in get_noise_generator(noise_params=noise_params, max_time=max_time):

            # Run the simulation on each budget
            for budget in sorted(budgets):
                print('===== Starting budget: {0:.4f} ====='.format(budget))

                result, noise_terms = run_simulation(runtime_systems=runtime_systems,
                                                     max_time=max_time,
                                                     noise_generator=noise_generator,
                                                     budget=budget)

                plot_and_save(sim_results=result,
                              runtime_systems=runtime_systems,
                              budget=budget,
                              max_time=max_time,
                              noise_generator=noise_generator,
                              noise_terms=noise_terms,
                              output_folder=args.output_folder,
                              should_plot=not args.skip_plotting,
                              save_plots=args.save_plots,
                              power_system_type=power_system_type,
                              baseline_to_plot=args.baseline_to_plot)
