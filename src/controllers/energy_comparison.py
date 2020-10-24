import os.path
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, Any, Tuple, List

from dataset.dataset import DataSeries
from dataset.dataset_factory import get_dataset
from utils.constants import SEQ_LENGTH, NUM_CLASSES
from utils.file_utils import read_by_file_suffix, save_by_file_suffix, extract_model_name
from utils.loading_utils import restore_neural_network
from noise_generators import get_noise_generator, NoiseGenerator
from model_controllers import AdaptiveController, CONTROLLER_PATH
from controller_utils import execute_adaptive_model, ModelResults
from runtime_system import SystemType, RuntimeSystem
from controllers.power_utils import make_power_system, PowerType


BudgetResult = namedtuple('BudgetResult', ['accuracy', 'budget'])
MAX_DIVISIONS = 5
MARGIN = 1e-4


def evaluate(system: RuntimeSystem, budget: float, max_time: int, noise_generator: NoiseGenerator) -> float:
    """
    Evaluates the adaptive system on the given budget.
    """
    system.init_for_budget(budget=budget, max_time=max_time)

    for t in range(max_time):
        power_noise = noise_generator.get_noise(t=t)
        system.step(budget=budget, power_noise=power_noise, t=t)

    accuracy = system.get_num_correct()[-1] / max_time
    return accuracy


def search_for_budget(test_results: ModelResults,
                      adaptive_system: RuntimeSystem,
                      lower_result: BudgetResult,
                      upper_result: BudgetResult,
                      target_accuracy: float,
                      noise_generator: NoiseGenerator) -> float:
    division_count = 0
    lower_budget = lower_result.budget
    upper_budget = upper_result.budget

    max_time = test_results.stop_probs.shape[0]

    budgets: List[float] = [lower_result.budget, upper_result.budget]
    accuracy_scores: List[float] = [lower_result.accuracy, upper_result.accuracy]

    accuracy = -1.0
    while division_count < MAX_DIVISIONS and abs(accuracy - target_accuracy) > MARGIN:

        budget = (lower_budget + upper_budget) / 2

        accuracy = evaluate(system=adaptive_system,
                            budget=budget,
                            max_time=max_time,
                            noise_generator=noise_generator)

        budgets.append(budget)
        accuracy_scores.append(accuracy)

        if accuracy > target_accuracy:
            upper_budget = budget
        else:
            lower_budget = budget

        division_count += 1

    idx = np.argmin(np.abs(np.array(accuracy_scores) - target_accuracy))
    budget = budgets[idx]
    accuracy = accuracy_scores[idx]

    return budget


def get_nearest_budgets(adaptive_log: Dict[str, Dict[str, Any]],
                        lowest_accuracy: float,
                        highest_accuracy: float,
                        target_accuracy: float,
                        sensor_type: str,
                        seq_length: int,
                        num_levels: int) -> Tuple[BudgetResult, BudgetResult, str]:
    """
    Returns the budgets which contain accuracy values bounding the given accuracy.

    Args:
        adaptive_log: The log containing results from the adaptive system
        accuracy: The accuracy to match
        seq_length: The original sequence length
    Returns:
        Tuple of three elements:
            (1) The lower bound budget and accuracy
            (2) The upper bound budget and accuracy
            (3) The name of the used system
    """
    budgets = np.array(list(map(float, adaptive_log.keys())))

    power_system = make_power_system(mode=PowerType[sensor_type.upper()],
                                     num_levels=num_levels,
                                     seq_length=seq_length)

    # Get the largest budget for which the accuracy is LESS than the target
    lower_budget = None
    lower_accuracy = None

    for budget_results in adaptive_log.values():
        budget = budget_results['BUDGET']
        accuracy = budget_results['ACCURACY']
        
        if accuracy < target_accuracy and (lower_budget is None or lower_budget < budget):
            lower_budget = budget
            lower_accuracy = accuracy

    # If there is no accuracy which is lower than the target,
    # then we return the lowest possible region.
    if lower_budget is None:
        min_power = power_system.get_min_power()
        min_result = BudgetResult(accuracy=lowest_accuracy, budget=min_power)

        min_budget = np.min(budgets)
        min_key = '{0:.4f}'.format(min_budget)
        name = adaptive_log[min_key]['ORIGINAL_SYSTEM_NAME']

        lower_accuracy = adaptive_log[min_key]['ACCURACY']
        lower_result = BudgetResult(accuracy=lower_accuracy, budget=min_budget)

        return min_result, lower_result, name

    # Get the smallest budget for which the accuracy is higher than the target
    upper_budget = None
    upper_accuracy = None
    
    for budget_results in adaptive_log.values():
        budget = budget_results['BUDGET']
        accuracy = budget_results['ACCURACY']
        
        if accuracy > target_accuracy and (upper_budget is None or budget < upper_budget):
            upper_budget = budget
            upper_accuracy = accuracy

    # If there is no accuracy that is greater than the target, then
    # we return the highest possible region
    if upper_budget is None:
        max_power = power_system.get_max_power()
        max_result = BudgetResult(accuracy=highest_accuracy, budget=max_power)

        max_budget = np.max(budgets)
        max_key = '{0:.4f}'.format(max_budget)
        name = adaptive_log[max_key]['ORIGINAL_SYSTEM_NAME']
        
        upper_accuracy = adaptive_log[max_key]['ACCURACY']
        upper_result = BudgetResult(accuracy=upper_accuracy, budget=max_budget)

        return upper_result, max_result, name

    upper_key = '{0:.4f}'.format(upper_budget)
    name = adaptive_log[upper_key]['ORIGINAL_SYSTEM_NAME']

    lower_result = BudgetResult(accuracy=lower_accuracy, budget=lower_budget)
    upper_result = BudgetResult(accuracy=upper_accuracy, budget=upper_budget)

    return lower_result, upper_result, name


def energy_comparison(adaptive_results: Dict[str, Dict[str, Any]],
                      baseline_results: Dict[str, Dict[str, Any]],
                      adaptive_system_dict: Dict[str, RuntimeSystem],
                      adaptive_result_dict: Dict[str, ModelResults],
                      seq_length: int,
                      num_levels: int,
                      sensor_type: str,
                      noise_generator: NoiseGenerator) -> List[float]:
    """
    Compare the energy results between the adaptive and baseline systems.
    """
    lowest_accuracy = min((result.accuracy[0] for result in adaptive_result_dict.values()))
    highest_accuracy = max((result.accuracy[-1] for result in adaptive_result_dict.values()))

    diff: List[float] = []

    for baseline_result in baseline_results.values():
        baseline_acc = baseline_result['ACCURACY']
        budget = baseline_result['BUDGET']

        lower, upper, name = get_nearest_budgets(adaptive_results,
                                                 target_accuracy=baseline_acc,
                                                 lowest_accuracy=lowest_accuracy,
                                                 highest_accuracy=highest_accuracy,
                                                 sensor_type=sensor_type,
                                                 seq_length=seq_length,
                                                 num_levels=num_levels)

        adaptive_budget = search_for_budget(adaptive_result_dict[name],
                                            adaptive_system=adaptive_system_dict[name],
                                            noise_generator=noise_generator,
                                            lower_result=lower,
                                            upper_result=upper,
                                            target_accuracy=baseline_acc)

        perc_diff = (budget - adaptive_budget) / adaptive_budget
        diff.append(perc_diff)

    return diff


def save(comparison: List[float], baseline_log_path: str):
    output_folder, baseline_file = os.path.split(baseline_log_path)

    # Read in the previous log if it exists
    comparison_log_path = os.path.join(output_folder, 'energy_comparison.jsonl.gz')

    if os.path.exists(comparison_log_path):
        comparison_log = list(read_by_file_suffix(comparison_log_path))[0]
    else:
        comparison_log = dict()

    # Save results under the baseline name
    tokens = baseline_file.split('-')
    policy = tokens[1].upper()
    model = tokens[2].upper()

    key = '{0} {1}'.format(model, policy)
    comparison_log[key] = {
        'mean': np.average(comparison),
        'std': np.std(comparison),
        'median': np.median(comparison),
        'raw': comparison
    }

    save_by_file_suffix([comparison_log], comparison_log_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--adaptive-model-paths', type=str, required=True, nargs='+')
    parser.add_argument('--adaptive-log', type=str, required=True)
    parser.add_argument('--baseline-logs', type=str, required=True, nargs='+')
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--noise-loc', type=float, default=0.0)
    parser.add_argument('--sensor-type', type=str, choices=['bluetooth', 'temp'], required=True)
    parser.add_argument('--should-print', action='store_true')
    args = parser.parse_args()

    # Load the target data-set
    dataset = get_dataset(dataset_type='standard', data_folder=args.dataset_folder)

    # Load the adaptive model results and controllers
    adaptive_result_dict: Dict[str, ModelResults] = dict()
    adaptive_system_dict: Dict[str, RuntimeSystem] = dict()

    for model_path in args.adaptive_model_paths:
        model, _ = restore_neural_network(model_path=model_path,
                                          dataset_folder=args.dataset_folder)
        seq_length = model.metadata[SEQ_LENGTH]
        num_classes = model.metadata[NUM_CLASSES]
        num_levels = model.num_outputs

        # Get the validation and test results
        valid_results = execute_adaptive_model(model=model,
                                               dataset=dataset,
                                               series=DataSeries.VALID)

        test_results = execute_adaptive_model(model=model,
                                              dataset=dataset,
                                              series=DataSeries.TEST)

        max_time = test_results.stop_probs.shape[0]

        # Make the run-time system
        system = RuntimeSystem(test_results=test_results,
                               valid_results=valid_results,
                               system_type=SystemType.ADAPTIVE,
                               model_path=model_path,
                               dataset_folder=args.dataset_folder,
                               power_system_type=PowerType[args.sensor_type.upper()],
                               seq_length=seq_length,
                               num_levels=num_levels,
                               num_classes=num_classes)

        key = 'SAMPLE_RNN({0}) ADAPTIVE'.format(model.stride_length)
        adaptive_result_dict[key] = test_results
        adaptive_system_dict[key] = system

    # Make the noise generator to get the log key
    noise_params = dict(noise_type='gaussian', loc=args.noise_loc, scale=0.01)
    noise_generator = list(get_noise_generator(noise_params, max_time=max_time))[0]
    noise_type = str(noise_generator)

    # Load the adaptive testing log
    adaptive_log = list(read_by_file_suffix(args.adaptive_log))[0]
    adaptive_results = adaptive_log[noise_type]

    for baseline_log_file in args.baseline_logs:
        # Load the baseline testing log
        baseline_log = list(read_by_file_suffix(baseline_log_file))[0]
        baseline_results = baseline_log[noise_type]

        if args.should_print:
            print(baseline_log_file)

        # Perform the comparison
        energy_diff = energy_comparison(adaptive_results=adaptive_results,
                                        baseline_results=baseline_results,
                                        adaptive_system_dict=adaptive_system_dict,
                                        adaptive_result_dict=adaptive_result_dict,
                                        seq_length=seq_length,
                                        num_levels=num_levels,
                                        sensor_type=args.sensor_type,
                                        noise_generator=noise_generator)

        # Save the results
        save(comparison=energy_diff, baseline_log_path=baseline_log_file)
