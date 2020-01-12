import re
import numpy as np
import os.path
from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from typing import Optional, Dict, Tuple, Any
from collections import namedtuple
from random import random

from dataset.dataset import DataSeries
from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset
from inference_policies import get_inference_policy
from recharge_estimators import get_recharge_estimator
from utils.hyperparameters import extract_hyperparameters
from plot_anytime_results import plot_energy, plot_levels


Bounds = namedtuple('Bounds', ['min', 'max'])


def extract_model_name(model_file: str) -> str:
    match = re.match(r'^model-([^\.]+)\.ckpt.*$', model_file)
    if not match:
        if model_file.startswith('model-'):
            return model_file[len('model-'):]
        return model_file
    return match.group(1)


def evaluate(model: RNNSampleModel,
             dataset: RNNSampleDataset,
             inference_period: float,
             energy_bounds: Bounds,
             recharge_estimator_params: Dict[str, Any],
             inference_policy_params: Dict[str, Any],
             charging_params: Dict[str, float],
             collection_params: Dict[str, float],
             processing_params: Dict[str, float],
             output_folder: RichPath,
             max_time: int):
    """
    Evaluates the anytime model in a simulated intermittent environment.

    Args:
        model: The trained RNN model
        dataset: The testing dataset
        inference_period: Desired sec / op to maintain
        energy_bounds: min/max energy
        recharge_estimator_params: Parameters for the recharge estimator
        inference_policy_params: Parameter for the inference policy
        charging_params: Parameters for charging
        collection_params: Parameters for data collection
        processing_params: Parameters for data processing
        output_folder: Folder in which to store results
        max_time: Time in which this experiment is run
    Returns:
        Nothing. Outputs are directly written to files in the output folder.
    """
    # Generate the test batches
    test_batch_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                       batch_size=1,
                                                       metadata=model.metadata,
                                                       should_shuffle=False,
                                                       drop_incomplete_batches=True)

    # Polices which govern the system
    inference_policy = get_inference_policy(inference_policy_params['name'],
                                            model.num_outputs,
                                            **inference_policy_params['params'])
    recharge_estimator = get_recharge_estimator(recharge_estimator_params['name'],
                                                **recharge_estimator_params['params'])

    # Initialize the system time and energy
    system_time = 0.0
    system_energy = energy_bounds.max
    num_inferences = 0.0

    # Dictionaries to store metrics
    energy_dict: Dict[float, float] = dict()  # time -> energy level
    error_dict: Dict[float, float] = dict()  # time -> mean squared error
    inference_dict: Dict[float, float] = dict()  # time -> inference rate
    selected_levels: List[int] = []

    # Initialize tracking
    energy_dict[0] = energy_bounds.max
    inference_dict[0] = 0

    for batch in test_batch_generator:
        
        # Exit if the experiment is over
        if system_time >= max_time:
            break

        # Tracks the time spent on this period
        period_time = 0.0

        # This tells us how many levels we need to collect / process
        num_levels = inference_policy.get_num_levels()
        selected_levels.append(num_levels)

        # Simulate the loss of energy for collection num_levels worth of data
        for _ in range(num_levels):
            collection_energy = max(collection_params['energy'] + np.random.normal(loc=0.0, scale=collection_params['energy_noise']), 0.0)
            collection_time = max(collection_params['time'] + np.random.normal(loc=0.0, scale=collection_params['time_noise']), 0.0)
 
            system_energy -= collection_energy
            period_time += collection_time

            # Log to energy dictionary
            energy_dict[system_time + period_time] = system_energy

        # Estimate the recharge rate
        estimated_recharge_rate = recharge_estimator.estimate()

        # Perform the inference
        feed_dict = model.batch_to_feed_dict(batch, is_train=False)
        inference_result, system_energy, period_time, computed_levels = model.anytime_inference(feed_dict=feed_dict,
                                                                                                max_num_levels=num_levels,
                                                                                                processing_params=processing_params,
                                                                                                system_energy=system_energy,
                                                                                                period_time=period_time,
                                                                                                inference_time=inference_period,
                                                                                                max_energy=energy_bounds.max,
                                                                                                min_energy=energy_bounds.min,
                                                                                                recharge_rate=estimated_recharge_rate)

        # Log results
        energy_dict[system_time + period_time] = system_energy

        # Log any obtained result
        if (computed_levels > 0):
            error_dict[system_time + period_time] = np.sum(np.square(inference_result - batch['output']))  # Might need to un-normalize the results
            num_inferences += 1

        # Recharge the system
        true_recharge_rate = max(charging_params['rate'] + np.random.normal(loc=0.0, scale=charging_params['noise']), 1e-7)
        
        energy_delta = energy_bounds.max - system_energy
        period_time += energy_delta / true_recharge_rate
        system_energy = energy_bounds.max

        # Log results
        system_time += period_time
        energy_dict[system_time] = system_energy
        inference_dict[system_time] = system_time / num_inferences if num_inferences > 0 else 0.0

        # Supply feedback
        recharge_estimator.update(true_recharge_rate)
        inference_policy.update(level=num_levels, reward=-1 * np.square(computed_levels - num_levels))

    # Save results
    output_folder.make_as_dir()
    energy_results_file = output_folder.join('energy_results.pkl.gz')
    energy_results_file.save_as_compressed_file(energy_dict)

    inference_results_file = output_folder.join('inference_results.pkl.gz')
    inference_results_file.save_as_compressed_file(inference_dict)

    # Call plotting scripts
    plot_energy(energy_data=energy_dict,
                inference_data=inference_dict,
                min_energy=energy_bounds.min,
                max_energy=energy_bounds.max,
                output_folder=output_folder)

    plot_levels(selected_levels, model.num_outputs, output_folder=output_folder)


def initialize_dataset(dataset_folder: str) -> RNNSampleDataset:
    """
    Initializes the dataset for testing.
    """
    dataset_folder_path = RichPath.create(dataset_folder)
    assert dataset_folder_path.exists()
    return RNNSampleDataset(train_folder=dataset_folder_path.join('train'),
                            valid_folder=dataset_folder_path.join('valid'),
                            test_folder=dataset_folder_path.join('test'))


def restore_model(model_path: str, model_params_file: str) -> RNNSampleModel:
    """
    Initializes the model and loads all trainable parameters.
    """
    hypers = extract_hyperparameters(model_params_file)[0]

    path_tokens = os.path.split(model_path)
    folder, file_name = path_tokens[0], path_tokens[1]

    model = RNNSampleModel(hypers, folder)
    model_name = extract_model_name(file_name)

    model.restore_parameters(model_name)
    model.make(is_train=False)
    model.restore_weights(model_name)
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--simulation-params', type=str, required=True)
    args = parser.parse_args()

    params_file = RichPath.create(args.simulation_params)
    assert params_file.exists(), f'The file {params_file} does not exist!'

    params_dict = params_file.read_by_file_suffix()
 
    dataset = initialize_dataset(params_dict['dataset_folder'])
    model = restore_model(model_path=params_dict['model_path'],
                          model_params_file=params_dict['model_params_file'])

    output_folder = RichPath.create(params_dict['output_folder'])

    # Conduct the evaluation
    evaluate(model=model,
             dataset=dataset,
             energy_bounds=Bounds(*params_dict['energy_bounds']),
             recharge_estimator_params=params_dict['recharge_estimator_params'],
             inference_policy_params=params_dict['inference_policy_params'],
             inference_period=params_dict['inference_period'],
             charging_params=params_dict['charging_params'],
             collection_params=params_dict['collection_params'],
             processing_params=params_dict['processing_params'],
             output_folder=output_folder,
             max_time=params_dict['max_time'])
