import os
import numpy as np
from argparse import ArgumentParser
from shutil import copyfile
from typing import List, Dict, Set, Any

from dataset.dataset import DataSeries
from utils.loading_utils import restore_neural_network
from utils.file_utils import read_by_file_suffix, save_by_file_suffix, extract_model_name, make_dir, iterate_files
from controllers.controller_utils import ModelResults, execute_adaptive_model, LOG_FILE_FMT, get_budget_index
from controllers.model_controllers import CONTROLLER_PATH, AdaptiveController


def get_avg_accuracy(validation_accuracy: Dict[float, float], budget: float) -> float:
    """
    Gets the average validation accuracy for the two budgets closest to the given budget.
    If the budget is out of the range of the dictionary, we return the accuracy
    of the nearest budget.
    """
    budget_under, budget_over = None, None
    for b in validation_accuracy.keys():
        if b < budget and (budget_under is None or b > budget_under):
            budget_under = b

        if b > budget and (budget_over is None or b < budget_over):
            budget_over = b

    if budget_under is None:
        assert budget_over is not None, 'Could to find bounds for {0}'.format(budget)
        return validation_accuracy[budget_over]
    elif budget_over is None:
        assert budget_under is not None, 'Could to find bounds for {0}'.format(budget)
        return validation_accuracy[budget_under]
    else:
        return (validation_accuracy[budget_under] + validation_accuracy[budget_over]) / 2


def get_model_for_budget(model_accuracy: List[Dict[float, float]], budget: float) -> int:
    best_idx, best_accuracy = 0, 0.0

    for model_idx, valid_accuracy in enumerate(model_accuracy):
        avg_acc = get_avg_accuracy(valid_accuracy, budget=budget)
        
        if best_accuracy < avg_acc:
            best_idx = model_idx
            best_accuracy = avg_acc

    return best_idx


def merge_simulation_logs(logs: List[Dict[str, Dict[str, Dict[str, Any]]]],
                          model_accuracy: List[Dict[float, float]],
                          system_type: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Merges the given adaptive logs according to the budget policy.
    """
    result: Dict[str, Dict[str, Dict[str, Any]]] = dict()
    for noise_type, noise_results in logs[0].items():
        result[noise_type] = dict()

        for budget in noise_results.keys():
            model_idx = get_model_for_budget(model_accuracy, budget=float(budget))

            log_result = logs[model_idx][noise_type][budget]
            log_result['ORIGINAL_SYSTEM_NAME'] = log_result['SYSTEM_NAME']
            log_result['SYSTEM_NAME'] = 'BUDGET_RNN {0}'.format(system_type.upper())

            result[noise_type][budget] = log_result

    return result


def merge_and_save(logs: List[Dict[str, Dict[str, Dict[str, Any]]]],
                   model_accuracy: List[Dict[float, float]],
                   output_folder: str,
                   system_name: str,
                   dataset_name: str,
                   power_system_type: str):
    merged_log = merge_simulation_logs(logs=logs,
                                       model_accuracy=model_accuracy,
                                       system_type=system_name)

    output_file_name = LOG_FILE_FMT.format(system_name, 'BUDGET_RNN-{0}-merged'.format(dataset_name), power_system_type)
    output_file = os.path.join(output_folder, output_file_name)
    save_by_file_suffix([merged_log], output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--models', type=str, required=True, nargs='+', help='Paths to the Budget RNNs. Can be a directory containing the models.')
    parser.add_argument('--dataset-folder', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--log-folder', type=str, required=True, help='Path to folder containing the simulation logs.')
    parser.add_argument('--power-system-type', type=str, choices=['bluetooth', 'temp'], required=True, help='The sensor type.')
    parser.add_argument('--output-folder', type=str, help='Path to the output folder.')
    args = parser.parse_args()

    output_folder = args.log_folder if args.output_folder is None else args.output_folder
    make_dir(output_folder)

    # We first copy all the non-SAMPLE models to the output folder. This is done for convenience
    for log_file_name in os.listdir(args.log_folder):
        if ('SAMPLE_RNN' not in log_file_name) or ('BUDGET_RNN' not in log_file_name):
            old_path = os.path.join(args.log_folder, log_file_name)
            new_path = os.path.join(output_folder, log_file_name)
            copyfile(old_path, new_path)

    # Restore the given models and get the validation results
    adaptive_model_accuracy: List[Dict[float, float]] = []
    fixed_model_accuracy: List[Dict[float, float]] = []

    adaptive_logs: List[Dict[str, Dict[str, Dict[str, Any]]]] = []
    fixed_budget_logs: List[Dict[str, Dict[str, Dict[str, Any]]]] = []
    fixed_acc_logs: List[Dict[str, Dict[str, Dict[str, Any]]]] = []
    randomized_logs: List[Dict[str, Dict[str, Dict[str, Any]]]] = []

    # Expand the model paths by unpacking directories
    model_paths: List[str] = []
    for model_path in args.models:
        if os.path.isdir(model_path):
            model_paths.extend(iterate_files(model_path, pattern=r'.*model-SAMPLE_RNN-.*'))
            model_paths.extend(iterate_files(model_path, pattern=r'.*model-BUDGET_RNN-.*'))
        else:
            model_paths.append(model_path)

    for model_idx, model_path in enumerate(model_paths):
        # Fetch the results on the validation set
        model, dataset = restore_neural_network(model_path, args.dataset_folder)
        valid_results = execute_adaptive_model(model, dataset, series=DataSeries.VALID) 

        # Create the adaptive controller
        save_folder, model_file_name = os.path.split(model_path)
        model_name = extract_model_name(model_file_name)
        controller_path = os.path.join(save_folder, CONTROLLER_PATH.format(args.power_system_type, model_name))

        if not os.path.exists(controller_path):
            continue

        print(model_path)
        print(controller_path)
        controller = AdaptiveController.load(save_file=controller_path,
                                             dataset_folder=args.dataset_folder,
                                             model_path=model_path)

        power_estimates = controller._power_system.get_power_estimates()

        if controller._est_accuracy is None:
            continue

        print('==========')

        adaptive_valid_accuracy: Dict[float, float] = dict()
        fixed_valid_accuracy: Dict[float, float] = dict()

        # Select budgets in which to use this model
        budgets = controller.budgets
        for budget in budgets:
            # Evaluate the adaptive system using the estimated results from threshold fitting
            adaptive_accuracy = controller.get_estimated_accuracy(budget=budget)

            print('Model Idx: {0}, Budget: {1}, Accuracy: {2:.4f}'.format(model_idx, budget, adaptive_accuracy))

            adaptive_valid_accuracy[budget] = adaptive_accuracy

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

            fixed_valid_accuracy[budget] = adjusted_fixed_accuracy
        
        adaptive_model_accuracy.append(adaptive_valid_accuracy)
        fixed_model_accuracy.append(fixed_valid_accuracy)

        # Get the simulation logs
        adaptive_log_file_name = LOG_FILE_FMT.format('adaptive', model_name, args.power_system_type)
        adaptive_log_path = os.path.join(args.log_folder, adaptive_log_file_name)
        adaptive_log = list(read_by_file_suffix(adaptive_log_path))[0]
        adaptive_logs.append(adaptive_log)

        fixed_budget_log_file_name = LOG_FILE_FMT.format('fixed_under_budget', model_name, args.power_system_type)
        fixed_budget_log_path = os.path.join(args.log_folder, fixed_budget_log_file_name)
        fixed_budget_log = list(read_by_file_suffix(fixed_budget_log_path))[0]
        fixed_budget_logs.append(fixed_budget_log)

        randomized_log_file_name = LOG_FILE_FMT.format('randomized', model_name, args.power_system_type)
        randomized_log_path = os.path.join(args.log_folder, randomized_log_file_name)
        randomized_log = list(read_by_file_suffix(randomized_log_path))[0]
        randomized_logs.append(randomized_log)

    # Merge the adaptive simulation results and save the result
    path_tokens = args.dataset_folder.split(os.sep)
    dataset_name = path_tokens[-2] if len(path_tokens[-1].strip()) > 0 else path_tokens[-3]

    merge_and_save(logs=adaptive_logs,
                   model_accuracy=adaptive_model_accuracy,
                   system_name='adaptive',
                   dataset_name=dataset_name,
                   output_folder=output_folder,
                   power_system_type=args.power_system_type)

    merge_and_save(logs=fixed_budget_logs,
                   model_accuracy=fixed_model_accuracy,
                   system_name='fixed_under_budget',
                   dataset_name=dataset_name,
                   output_folder=output_folder,
                   power_system_type=args.power_system_type)

    merge_and_save(logs=randomized_logs,
                   model_accuracy=adaptive_model_accuracy,
                   system_name='randomized',
                   dataset_name=dataset_name,
                   output_folder=output_folder,
                   power_system_type=args.power_system_type)
