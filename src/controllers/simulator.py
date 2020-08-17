import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import time
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from scipy import integrate
from typing import Tuple, List, Union, Optional, Dict, Any

from controllers.runtime_system import RuntimeSystem, SystemType
from controllers.controller_utils import execute_adaptive_model, execute_standard_model
from controllers.controller_utils import save_test_log, execute_skip_rnn_model, ModelResults
from controllers.noise_generators import get_noise_generator, NoiseGenerator
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


LOG_FILE_FMT = 'model-{0}-{1}.jsonl.gz'
SimulationResult = namedtuple('SimulationResult', ['accuracy', 'power', 'target_budgets'])


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
        system_result = SimulationResult(power=system.get_energy() / times,
                                         accuracy=system.get_num_correct() / times,
                                         target_budgets=system.get_target_budgets())
        result[system.name] = system_result

    return result, noise_terms


def plot_and_save(sim_results: Dict[str, SimulationResult],
                  runtime_systems: List[RuntimeSystem],
                  output_folder: str,
                  budget: float,
                  max_time: int,
                  noise_generator: NoiseGenerator,
                  noise_terms: List[float],
                  should_plot: bool,
                  save_plots: bool):
    # Make the output folder if necessary
    make_dir(output_folder)

    # Log the test results for each adaptive system
    system_dict = {system.name: system for system in runtime_systems}
    for system_name in sorted(sim_results.keys()):
        system = system_dict[system_name]
        sim_result = sim_results[system_name]

        # We compute the validation accuracy for this budget for the adaptive models.
        # This allows us to choose which backend model to select at testing time.
        if system.system_type == SystemType.ADAPTIVE:
            valid_accuracy = system.estimate_validation_accuracy(budget=budget + noise_generator.loc)
        else:
            valid_accuracy = None

        log_file_name = LOG_FILE_FMT.format(system.system_type.name.lower(), system.model_name)
        log_path = os.path.join(output_folder, log_file_name)
        save_test_log(accuracy=sim_result.accuracy[-1],
                      power=sim_result.power[-1],
                      valid_accuracy=valid_accuracy,
                      budget=budget,
                      key=str(noise_generator),
                      output_file=log_path)

        print('{0} Accuracy: {1:.5f}, {0} Power: {2:.5f}'.format(system_name, sim_result.accuracy[-1], sim_result.power[-1]))

    if not should_plot:
        return

    # List of times for plotting
    times = np.arange(max_time) + 1

    # Plot the results
    with plt.style.context('ggplot'):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(16, 12), nrows=4, ncols=1, sharex=True)

        # Plot the power noise terms
        ax1.plot(times, noise_terms)
        ax1.set_title('Power Noise')
        ax1.set_ylabel('Power (mw)')

        # Plot the Setpoints of each system
        for system_name, sim_result in sorted(sim_results.items()):
            ax2.plot(times, sim_result.target_budgets, label=system_name)
        ax2.legend()
        ax2.set_title('Target Power Setpoint for Each Policy')
        ax2.set_ylabel('Power (mW)')

        # Plot the accuracy of each system
        for system_name, sim_result in sorted(sim_results.items()):
            ax3.plot(times, sim_result.accuracy, label=system_name)
        ax3.legend()
        ax3.set_title('Cumulative Accuracy for Each Policy')
        ax3.set_ylabel('Accuracy')

        # Plot the Avg Power of each system
        for system_name, sim_result in sorted(sim_results.items()):
            ax4.plot(times, sim_result.power, label=system_name)
        power_budget = [budget for _ in times]
        ax4.plot(times, power_budget, label='Budget')

        ax4.legend()
        ax4.set_title('Cumulative Avg Power for Each Policy')
        ax4.set_ylabel('Power (mW)')
        ax4.set_xlabel('Time')

        plt.tight_layout()

        if save_plots:
            output_file = os.path.join(output_folder, 'results_{0}.pdf'.format(budget))
            plt.savefig(output_file)
        else:
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--adaptive-model-paths', type=str, nargs='+', required=True)
    parser.add_argument('--baseline-model-path', type=str, required=True)
    parser.add_argument('--skip-model-folder', type=str)
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--budget-start', type=float, required=True)
    parser.add_argument('--budget-end', type=float, required=True)
    parser.add_argument('--budget-step', type=float, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--noise-params', type=str, nargs='+', required=True)
    parser.add_argument('--skip-plotting', action='store_true')
    parser.add_argument('--save-plots', action='store_true')
    args = parser.parse_args()

    # Validate arguments
    budget_start, budget_end, budget_step = args.budget_start, args.budget_end, args.budget_step
    assert budget_start > 0, 'Must have a positive budget'
    assert budget_end >= budget_start, 'Must have budget_end >= budget_start'
    assert budget_step > 0, 'Must have a positive budget step'

    budgets = np.arange(start=budget_start, stop=budget_end + (budget_step / 2), step=budget_step)

    dataset_folder = args.dataset_folder

    # Make systems based on adaptive models
    runtime_systems: List[RuntimeSystem] = []
    for adaptive_model_path in args.adaptive_model_paths:
        model, dataset = restore_neural_network(adaptive_model_path, dataset_folder=dataset_folder)

        num_levels = model.num_outputs
        seq_length = model.metadata[SEQ_LENGTH]
        num_classes = model.metadata[NUM_CLASSES]

        valid_results = execute_adaptive_model(model, dataset, series=DataSeries.VALID)
        test_results = execute_adaptive_model(model, dataset, series=DataSeries.TEST)

        adaptive_system = RuntimeSystem(valid_results=valid_results,
                                        test_results=test_results,
                                        system_type=SystemType.ADAPTIVE,
                                        model_path=adaptive_model_path,
                                        dataset_folder=dataset_folder,
                                        seq_length=seq_length,
                                        num_levels=num_levels,
                                        num_classes=num_classes)
        runtime_systems.append(adaptive_system)

        adaptive_fixed_under_budget = RuntimeSystem(valid_results=valid_results,
                                                    test_results=test_results,
                                                    system_type=SystemType.FIXED_UNDER_BUDGET,
                                                    model_path=adaptive_model_path,
                                                    dataset_folder=dataset_folder,
                                                    seq_length=seq_length,
                                                    num_levels=num_levels,
                                                    num_classes=num_classes)
        runtime_systems.append(adaptive_fixed_under_budget)

        adaptive_fixed_max_accuracy = RuntimeSystem(valid_results=valid_results,
                                                    test_results=test_results,
                                                    system_type=SystemType.FIXED_MAX_ACCURACY,
                                                    model_path=adaptive_model_path,
                                                    dataset_folder=dataset_folder,
                                                    seq_length=seq_length,
                                                    num_levels=num_levels,
                                                    num_classes=num_classes)
        runtime_systems.append(adaptive_fixed_max_accuracy)

        randomized_system = RuntimeSystem(valid_results=valid_results,
                                          test_results=test_results,
                                          system_type=SystemType.RANDOMIZED,
                                          model_path=adaptive_model_path,
                                          dataset_folder=dataset_folder,
                                          seq_length=seq_length,
                                          num_levels=num_levels,
                                          num_classes=num_classes)
        runtime_systems.append(randomized_system)

    # Make the baseline systems
    baseline_model_path = args.baseline_model_path
    model, dataset = restore_neural_network(baseline_model_path, dataset_folder=dataset_folder)

    valid_results = execute_standard_model(model, dataset, series=DataSeries.VALID)
    test_results = execute_standard_model(model, dataset, series=DataSeries.TEST)

    seq_length = model.metadata[SEQ_LENGTH]
    num_classes = model.metadata[NUM_CLASSES]

    greedy_system = RuntimeSystem(test_results=test_results,
                                  valid_results=valid_results,
                                  system_type=SystemType.GREEDY,
                                  model_path=baseline_model_path,
                                  dataset_folder=dataset_folder,
                                  seq_length=seq_length,
                                  num_levels=seq_length,
                                  num_classes=num_classes)
    runtime_systems.append(greedy_system)

    fixed_under_budget = RuntimeSystem(test_results=test_results,
                                       valid_results=valid_results,
                                       system_type=SystemType.FIXED_UNDER_BUDGET,
                                       model_path=baseline_model_path,
                                       dataset_folder=dataset_folder,
                                       seq_length=seq_length,
                                       num_levels=seq_length,
                                       num_classes=num_classes)
    runtime_systems.append(fixed_under_budget)

    fixed_max_accuracy = RuntimeSystem(test_results=test_results,
                                       valid_results=valid_results,
                                       system_type=SystemType.FIXED_MAX_ACCURACY,
                                       model_path=baseline_model_path,
                                       dataset_folder=dataset_folder,
                                       seq_length=seq_length,
                                       num_levels=seq_length,
                                       num_classes=num_classes)
    runtime_systems.append(fixed_max_accuracy)

    # Add the Skip RNN models if provided
    skip_rnn_folder = args.skip_model_folder
    if skip_rnn_folder is not None:
        valid_results: List[ModelResults] = []
        test_results: List[ModelResults] = []
        model_paths: List[str] = []
        for model_path in iterate_files(skip_rnn_folder, pattern='model-SKIP_RNN-.*model_best\.pkl\.gz'):
            model, dataset = restore_neural_network(model_path, dataset_folder=dataset_folder)

            valid_result = execute_skip_rnn_model(model, dataset, series=DataSeries.VALID)
            test_result = execute_skip_rnn_model(model, dataset, series=DataSeries.TEST)

            valid_results.append(valid_result)
            test_results.append(test_result)

            model_paths.append(model_path)

        # Concatenate the results from each model
        valid_predictions = np.concatenate([r.predictions for r in valid_results], axis=1)  # [N, L]
        valid_labels = valid_results[0].labels  # [N, 1]
        valid_stop_probs = [r.stop_probs for r in valid_results]
        valid_accuracy = [r.accuracy for r in valid_results]
        valid_skip_results = ModelResults(predictions=valid_predictions, labels=valid_labels, stop_probs=valid_stop_probs, accuracy=valid_accuracy)

        test_predictions = np.concatenate([r.predictions for r in test_results], axis=1)  # [N, L]
        test_labels = test_results[0].labels  # [N, 1]
        test_stop_probs = [r.stop_probs for r in test_results]
        test_accuracy = [r.accuracy for r in test_results]
        test_skip_results = ModelResults(predictions=test_predictions, labels=test_labels, stop_probs=test_stop_probs, accuracy=test_accuracy)

        skip_under_budget = RuntimeSystem(test_results=test_skip_results,
                                          valid_results=valid_skip_results,
                                          system_type=SystemType.FIXED_UNDER_BUDGET,
                                          model_path=model_paths[0],  # Design decision: Pick the first model path to save results under
                                          dataset_folder=dataset_folder,
                                          seq_length=seq_length,
                                          num_levels=len(model_paths),
                                          num_classes=num_classes)
        runtime_systems.append(skip_under_budget)

        skip_max_accuracy = RuntimeSystem(test_results=test_skip_results,
                                          valid_results=valid_skip_results,
                                          system_type=SystemType.FIXED_MAX_ACCURACY,
                                          model_path=model_paths[0],  # Design decision: Pick the first model path to save results under
                                          dataset_folder=dataset_folder,
                                          seq_length=seq_length,
                                          num_levels=len(model_paths),
                                          num_classes=num_classes)
        runtime_systems.append(skip_max_accuracy)

    # Max time equals the number of test samples
    max_time = dataset.dataset[DataSeries.TEST].length

    for noise_params_path in args.noise_params:
        noise_params = read_by_file_suffix(noise_params_path)

        # Create the noise generator for the given parameters
        for noise_generator in get_noise_generator(noise_params=noise_params, max_time=max_time):

            # Run the simulation on each budget
            for budget in sorted(budgets):
                print('===== Starting budget: {0} ====='.format(budget))

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
                              save_plots=args.save_plots)
