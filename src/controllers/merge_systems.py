import os
import numpy as np
from argparse import ArgumentParser
from shutil import copyfile
from typing import List, Dict, Set, Any

from dataset.dataset import DataSeries
from utils.loading_utils import restore_neural_network
from utils.file_utils import read_by_file_suffix, save_by_file_suffix, extract_model_name, make_dir
from controllers.controller_utils import ModelResults, execute_adaptive_model, LOG_FILE_FMT, get_budget_index
from controllers.model_controllers import CONTROLLER_PATH, AdaptiveController
from controllers.power_utils import get_power_estimates


def merge_simulation_logs(logs: List[Dict[str, Dict[str, Dict[str, Any]]]],
                          budget_policy: Dict[float, int],
                          system_type: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Merges the given adaptive logs according to the budget policy.
    """
    budgets = np.array(list(sorted(budget_policy.keys())))

    result: Dict[str, Dict[str, Dict[str, Any]]] = dict()
    for noise_type, noise_results in logs[0].items():
        result[noise_type] = dict()

        for budget in noise_results.keys():
            # Get the model for the closest budget
            nearest_budget_idx = np.argmin(np.abs(float(budget) - budgets))
            nearest_budget = budgets[nearest_budget_idx]
            model_idx = budget_policy[nearest_budget]

            log_result = logs[model_idx][noise_type][budget]
            log_result['ORIGINAL_SYSTEM_NAME'] = log_result['SYSTEM_NAME']
            log_result['SYSTEM_NAME'] = 'SAMPLE_RNN {0}'.format(system_type.upper())

            result[noise_type][budget] = log_result

    return result


def merge_and_save(logs: List[Dict[str, Dict[str, Dict[str, Any]]]],
                   budget_policy: Dict[float, int],
                   output_folder: str,
                   system_name: str,
                   dataset_name: str):
    merged_log = merge_simulation_logs(logs=logs,
                                       budget_policy=budget_policy,
                                       system_type=system_name)

    output_file_name = LOG_FILE_FMT.format(system_name, 'SAMPLE_RNN-{0}-merged'.format(dataset_name))
    output_file = os.path.join(output_folder, output_file_name)
    save_by_file_suffix([merged_log], output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--models', type=str, required=True, nargs='+')
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--log-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str)
    args = parser.parse_args()

    output_folder = args.log_folder if args.output_folder is None else args.output_folder
    make_dir(output_folder)

    # We first copy all the non-SAMPLE models to the output folder. This is done for convenience
    for log_file_name in os.listdir(args.log_folder):
        if 'SAMPLE_RNN' not in log_file_name:
            old_path = os.path.join(args.log_folder, log_file_name)
            new_path = os.path.join(output_folder, log_file_name)
            copyfile(old_path, new_path)

    # Restore the given models and get the validation results
    adaptive_policy: Dict[float, int] = dict()  # Maps budget to corresponding model index (for the adaptive policy)
    fixed_policy: Dict[float, int] = dict()  # Maps budget to corresponding model index (for the fixed policy)

    adaptive_best_accuracy: Dict[float, float] = dict()  # Maps budget to the best validation accuracy
    fixed_best_accuracy: Dict[float, float] = dict()

    best_power: Dict[float, float] = dict()

    adaptive_logs: List[Dict[str, Dict[str, Dict[str, Any]]]] = []
    fixed_budget_logs: List[Dict[str, Dict[str, Dict[str, Any]]]] = []
    fixed_acc_logs: List[Dict[str, Dict[str, Dict[str, Any]]]] = []
    randomized_logs: List[Dict[str, Dict[str, Dict[str, Any]]]] = []

    for model_idx, model_path in enumerate(args.models):
        # Fetch the results on the validation set
        model, dataset = restore_neural_network(model_path, args.dataset_folder)
        valid_results = execute_adaptive_model(model, dataset, series=DataSeries.VALID) 

        # Estimate power at each level
        power_estimates = get_power_estimates(num_levels=model.num_outputs, seq_length=model.seq_length)

        # Create the adaptive controller
        save_folder, model_file_name = os.path.split(model_path)
        model_name = extract_model_name(model_file_name)
        controller_path = os.path.join(save_folder, CONTROLLER_PATH.format(model_name))

        controller = AdaptiveController.load(save_file=controller_path,
                                             dataset_folder=args.dataset_folder,
                                             model_path=model_path)

        print('==========')

        # Select budgets in which to use this model
        budgets = controller.budgets
        for budget in budgets:
            
            # Evaluate the adaptive system
            adaptive_accuracy, _ = controller.evaluate(budget=budget, model_results=valid_results)
            
            if adaptive_best_accuracy.get(budget, 0.0) < adaptive_accuracy:
                adaptive_policy[budget] = model_idx
                adaptive_best_accuracy[budget] = adaptive_accuracy

            # Evaluate the fixed system
            max_time = valid_results.stop_probs.shape[0]
            level = get_budget_index(budget=budget,
                                     valid_accuracy=valid_results.accuracy,
                                     max_time=max_time,
                                     power_estimates=power_estimates,
                                     allow_violations=True)
            fixed_accuracy = valid_results.accuracy[level]
            fixed_power = power_estimates[level]

            # Adjust accuracy based on budget
            time_steps = np.minimum(((budget * max_time) / fixed_power).astype(int), max_time)  # [S]
            adjusted_fixed_accuracy = (fixed_accuracy * time_steps) / max_time  # [S]

            if fixed_best_accuracy.get(budget, 0.0) < adjusted_fixed_accuracy:
                fixed_policy[budget] = model_idx
                fixed_best_accuracy[budget] = adjusted_fixed_accuracy

        # Get the simulation logs
        adaptive_log_file_name = LOG_FILE_FMT.format('adaptive', model_name)
        adaptive_log_path = os.path.join(args.log_folder, adaptive_log_file_name)
        adaptive_log = list(read_by_file_suffix(adaptive_log_path))[0]
        adaptive_logs.append(adaptive_log)

        fixed_acc_log_file_name = LOG_FILE_FMT.format('fixed_max_accuracy', model_name)
        fixed_acc_log_path = os.path.join(args.log_folder, fixed_acc_log_file_name)
        fixed_acc_log = list(read_by_file_suffix(fixed_acc_log_path))[0]
        fixed_acc_logs.append(fixed_acc_log)

        fixed_budget_log_file_name = LOG_FILE_FMT.format('fixed_under_budget', model_name)
        fixed_budget_log_path = os.path.join(args.log_folder, fixed_budget_log_file_name)
        fixed_budget_log = list(read_by_file_suffix(fixed_budget_log_path))[0]
        fixed_budget_logs.append(fixed_budget_log)

        randomized_log_file_name = LOG_FILE_FMT.format('randomized', model_name)
        randomized_log_path = os.path.join(args.log_folder, randomized_log_file_name)
        randomized_log = list(read_by_file_suffix(randomized_log_path))[0]
        randomized_logs.append(randomized_log)

    # Merge the adaptive simulation results and save the result
    path_tokens = args.dataset_folder.split(os.sep)
    dataset_name = path_tokens[-2] if len(path_tokens[-1].strip()) > 0 else path_tokens[-3]

    merge_and_save(logs=adaptive_logs,
                   budget_policy=adaptive_policy,
                   system_name='adaptive',
                   dataset_name=dataset_name,
                   output_folder=output_folder)

    merge_and_save(logs=fixed_acc_logs,
                   budget_policy=fixed_policy,
                   system_name='fixed_max_accuracy',
                   dataset_name=dataset_name,
                   output_folder=output_folder)

    merge_and_save(logs=fixed_budget_logs,
                   budget_policy=fixed_policy,
                   system_name='fixed_under_budget',
                   dataset_name=dataset_name,
                   output_folder=output_folder)

    merge_and_save(logs=randomized_logs,
                   budget_policy=adaptive_policy,
                   system_name='randomized',
                   dataset_name=dataset_name,
                   output_folder=output_folder)
