import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import time
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from scipy import integrate
from typing import Tuple, List, Union, Optional, Dict

from controllers.runtime_system import RuntimeSystem
from controllers.controller_utils import interpolate_power, get_power_for_levels, POWER, execute_adaptive_model, execute_standard_model
from controllers.controller_utils import save_test_log
from models.base_model import Model
from models.model_factory import get_model
from models.adaptive_model import AdaptiveModel
from models.standard_model import StandardModel
from dataset.dataset import DataSeries, Dataset
from dataset.dataset_factory import get_dataset
from utils.hyperparameters import HyperParameters
from utils.file_utils import extract_model_name, read_by_file_suffix, save_by_file_suffix, make_dir
from utils.constants import SMALL_NUMBER, METADATA_PATH, HYPERS_PATH, SEQ_LENGTH, NUM_CLASSES


LOG_FILE_FMT = 'model-{0}-{1}.jsonl.gz'
SimulationResult = namedtuple('SimulationResult', ['accuracy', 'power', 'target_budgets'])



def make_dataset(model_name: str, save_folder: str, dataset_type: str, dataset_folder: Optional[str]) -> Dataset:
    metadata_file = os.path.join(save_folder, METADATA_PATH.format(model_name))
    metadata = read_by_file_suffix(metadata_file)

    # Infer the dataset
    if dataset_folder is None:
        dataset_folder = os.path.dirname(metadata['data_folders'][TRAIN.upper()])

    # Validate the dataset folder
    assert os.path.exists(dataset_folder), f'The dataset folder {dataset_folder} does not exist!'

    return get_dataset(dataset_type=dataset_type, data_folder=dataset_folder)


def make_model(model_name: str, hypers: HyperParameters, save_folder: str) -> Model:
    model = get_model(hypers, save_folder, is_train=False)
    model.restore(name=model_name, is_train=False, is_frozen=False)
    return model


def get_serialized_info(model_path: str, dataset_folder: Optional[str]) -> Tuple[AdaptiveModel, Dataset]:
    save_folder, model_file = os.path.split(model_path)

    model_name = extract_model_name(model_file)
    assert model_name is not None, f'Could not extract name from file: {model_file}'

    # Extract hyperparameters
    hypers_path = os.path.join(save_folder, HYPERS_PATH.format(model_name))
    hypers = HyperParameters.create_from_file(hypers_path)

    dataset = make_dataset(model_name, save_folder, hypers.dataset_type, dataset_folder)
    model = make_model(model_name, hypers, save_folder)

    return model, dataset


def run_simulation(runtime_systems: List[RuntimeSystem], budget: float, noise: Tuple[float, float], max_time: int) -> SimulationResult:

    # Initialize the systems for this budget
    for system in runtime_systems:
        system.init_for_budget(budget=budget, max_time=max_time)

    # Set random state for reproducible results
    rand = np.random.RandomState(seed=42)
    power_noise = rand.normal(loc=noise[0], scale=noise[1], size=(max_time, ))

    # Sequentially execute each system
    for t in range(max_time):
        for system in runtime_systems:
            system.step(budget=budget, power_noise=power_noise[t], time=t)

    # Create the final result
    times = np.arange(max_time) + 1
    result: Dict[str, SimulationResult] = dict()

    for system in runtime_systems:
        system_result = SimulationResult(power=system.get_energy() / times,
                                         accuracy=system.get_num_correct() / times,
                                         target_budgets=system.get_target_budgets())
        result[system.name] = system_result

    return result


def plot_and_save(sim_results: Dict[str, SimulationResult],
                  runtime_systems: List[RuntimeSystem],
                  output_folder: Optional[str],
                  budget: int,
                  max_time: int,
                  noise_loc: float,
                  should_plot: bool):
    # Log the test results for each adaptive system
    system_dict = {system.name: system for system in runtime_systems}
    for system_name in sim_results.keys():
        system = system_dict[system_name]
        sim_result = sim_results[system_name]

        log_file_name = LOG_FILE_FMT.format(system.system_type.name.lower(), system.model_name)
        log_path = os.path.join(system.save_folder, log_file_name)
        # save_test_log(sim_result.accuracy[-1], sim_result.power[-1], budget, noise_loc, log_path)

        print('{0} Accuracy: {1:.5f}, {0} Power: {2:.5f}'.format(system.system_type.name.capitalize(), sim_result.accuracy[-1], sim_result.power[-1]))

    if not should_plot:
        return

    # List of times for plotting
    times = np.arange(max_time) + 1

    # Plot the results
    with plt.style.context('ggplot'):
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(16, 12), nrows=3, ncols=1, sharex=True)

        # Plot the Setpoints of each system
        for system_name, sim_result in sorted(sim_results.items()):
            ax1.plot(times, sim_result.target_budgets, label=system_name)
        ax1.legend()
        ax1.set_title('Target Power Setpoint for Each Policy')
        ax1.set_ylabel('Power (mW)')

        # Plot the accuracy of each system
        for system_name, sim_result in sorted(sim_results.items()):
            ax2.plot(times, sim_result.accuracy, label=system_name)
        ax2.legend()
        ax2.set_title('Cumulative Accuracy for Each Policy')
        ax2.set_ylabel('Accuracy')

        # Plot the Avg Power of each system
        for system_name, sim_result in sorted(sim_results.items()):
            ax3.plot(times, sim_result.power, label=system_name)
        power_budget = [budget for _ in times]
        ax3.plot(times, power_budget, label='Budget')

        ax3.legend()
        ax3.set_title('Cumulative Avg Power for Each Policy')
        ax3.set_ylabel('Power (mW)')
        ax3.set_xlabel('Time')

        plt.tight_layout()

        if output_folder is not None:
            make_dir(output_folder)
            output_file = os.path.join(output_folder, 'results_{0}.pdf'.format(budget))
            plt.savefig(output_file)
        else:
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--adaptive-model-paths', type=str, nargs='+', required=True)
    parser.add_argument('--baseline-model-path', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--budgets', type=float, nargs='+')
    parser.add_argument('--output-folder', type=str)
    parser.add_argument('--noise-loc', type=float, default=0.0)
    parser.add_argument('--noise-scale', type=float, default=1.0)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--skip-plotting', action='store_true')
    args = parser.parse_args()

    # Validate arguments
    budgets = args.budgets
    assert all([b > 0 for b in budgets]), 'Must have a positive budgets'
    assert args.noise_scale > 0, 'Must have a positive noise scale'

    dataset_folder = args.dataset_folder

    # Make systems based on adaptive models
    runtime_systems: List[RuntimeSystem] = []
    for adaptive_model_path in args.adaptive_model_paths:
        model, dataset = get_serialized_info(adaptive_model_path, dataset_folder=dataset_folder)
        num_levels = model.num_outputs
        power_estimates = get_power_for_levels(POWER, num_levels=num_levels)

        model_results = execute_adaptive_model(model, dataset, series=DataSeries.TEST)

        adaptive_system = RuntimeSystem(model_results=model_results,
                                        system_type='adaptive',
                                        model_path=adaptive_model_path,
                                        dataset_folder=dataset_folder,
                                        power_estimates=power_estimates,
                                        num_levels=num_levels,
                                        num_classes=model.metadata[NUM_CLASSES])
        runtime_systems.append(adaptive_system)

        randomized_system = RuntimeSystem(model_results=model_results,
                                          system_type='randomized',
                                          model_path=adaptive_model_path,
                                          dataset_folder=dataset_folder,
                                          power_estimates=power_estimates,
                                          num_levels=num_levels,
                                          num_classes=model.metadata[NUM_CLASSES])
        runtime_systems.append(randomized_system)

    # Make the baseline systems
    baseline_model_path = args.baseline_model_path
    model, dataset = get_serialized_info(baseline_model_path, dataset_folder=dataset_folder)

    model_results = execute_standard_model(model, dataset, series=DataSeries.TEST)

    seq_length = model.metadata[SEQ_LENGTH]
    power_estimates = get_power_for_levels(POWER, num_levels=num_levels)
    power_estimates = interpolate_power(power_estimates, seq_length)

    greedy_system = RuntimeSystem(model_results=model_results,
                                  system_type='greedy',
                                  model_path=baseline_model_path,
                                  dataset_folder=dataset_folder,
                                  power_estimates=power_estimates,
                                  num_levels=seq_length,
                                  num_classes=model.metadata[NUM_CLASSES])
    runtime_systems.append(greedy_system)

    fixed_system = RuntimeSystem(model_results=model_results,
                                 system_type='fixed',
                                 model_path=baseline_model_path,
                                 dataset_folder=dataset_folder,
                                 power_estimates=power_estimates,
                                 num_levels=seq_length,
                                 num_classes=model.metadata[NUM_CLASSES])
    runtime_systems.append(fixed_system)

    # Max time equals the number of test samples
    max_time = dataset.dataset[DataSeries.TEST].length

    # Run the simulation on each budget
    for budget in sorted(args.budgets):
        print('Starting budget: {0}'.format(budget))

        result = run_simulation(runtime_systems=runtime_systems, 
                                noise=(args.noise_loc, args.noise_scale),
                                max_time=max_time,
                                budget=budget)

        plot_and_save(sim_results=result,
                      runtime_systems=runtime_systems,
                      budget=budget,
                      max_time=max_time,
                      noise_loc=args.noise_loc,
                      output_folder=args.output_folder,
                      should_plot=not args.skip_plotting)