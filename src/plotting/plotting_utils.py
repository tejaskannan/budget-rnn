import re
import numpy as np
from collections import defaultdict, namedtuple
from typing import Dict, DefaultDict, List, Optional, Tuple

from utils.file_utils import iterate_files, read_by_file_suffix
from utils.constants import SMALL_NUMBER, BIG_NUMBER
from controllers.noise_generators import get_noise_generator, NoiseGenerator

ModelResult = namedtuple('ModelResult', ['power', 'accuracy', 'validation_accuracy'])

MODEL_REGEX = re.compile('.*model-([^-]+)-([^-]+)-([^0-9]+)-.*jsonl\.gz')

BASELINE_MODELS = ['RNN', 'NBOW']
ADAPTIVE_MODELS = ['SAMPLE_RNN', 'SAMPLE_NBOW', 'CASCADE_RNN', 'CASCADE_NBOW']

DATASET_MAP = {
    'emg': 'EMG',
    'forda': 'Ford A',
    'melbourne-pedestrian': 'Pedestrian',
    'pavement': 'Pavement',
    'pen-digits': 'Pen Digits',
    'whale': 'Whale',
    'uci-har': 'UCI HAR'
}


def rename_system(system_name: str) -> str:
    if 'CASCADE' in system_name:
        return system_name.replace('CASCADE', 'SAMPLE')
    return system_name


def rename_dataset(dataset_name: str) -> str:
    return DATASET_MAP[dataset_name.lower()]


def to_label(name: str) -> str:
    space_separated = name.replace('_', ' ').replace('-', ' ')
    tokens = space_separated.split()
    return ' '.join((t.capitalize() if t.lower() != 'rnn' else t.upper() for t in tokens))


def get_model_and_type(path: str) -> Optional[Tuple[str, str, str]]:
    match = MODEL_REGEX.match(path)
    if match is None:
        return None

    return match.group(1), match.group(2), match.group(3)


def make_noise_generator(noise_type: str,
                         noise_loc: float,
                         noise_scale: float,
                         noise_period: Optional[int],
                         noise_amplitude: Optional[float]) -> NoiseGenerator:
    noise_params = dict(loc=noise_loc, scale=noise_scale, period=noise_period, amplitude=noise_amplitude, noise_type=noise_type)
    return list(get_noise_generator(noise_params=noise_params, max_time=0))[0]


def select_adaptive_system(model_results: Dict[float, Dict[str, List[ModelResult]]], baseline_name: str) -> str:
    average_results: DefaultDict[str, List[float]] = defaultdict(list)
    for _, results in model_results.items():

        for model_name, system_results in results.items():
            if not model_name.startswith('ADAPTIVE'):
                continue

            acc = np.average([r.accuracy for r in system_results])
            average_results[model_name].append(acc)

    best_model = None
    best_accuracy = -BIG_NUMBER
    for name, accuracy in average_results.items():
        avg_accuracy = np.average(accuracy)

        if avg_accuracy > best_accuracy:
            best_model = name[len('ADAPTIVE') + 1:]
            best_accuracy = avg_accuracy

    return best_model


def get_results(input_folders: List[str], noise_generator: NoiseGenerator, model_type: str, baseline_mode: str) -> Dict[str, DefaultDict[float, Dict[str, List[ModelResult]]]]:
    """
    Gets the results for all models in the given folder with the given power shift value.

    Args:
        input_folders: A list of input folders containing model results.
        target_shift: The power shift to extract results from
    Returns:
        A dictionary of the following format.
        Key: Dataset Name
        Value: A dictionary of the format below.
            Key: Budget
            Value: Dictionary of Model Name -> List of accuracy values.
    """
    # Create the key for this series
    noise_key = str(noise_generator)
    fixed_type = 'fixed_{0}'.format(baseline_mode)
   
    model_results: Dict[str, DefaultDict[float, Dict[str, List[float]]]] = dict()

    for folder in input_folders:

        for file_name in iterate_files(folder, pattern=r'.*\.jsonl\.gz'):

            model_info = get_model_and_type(file_name)
            if model_info is None:
                continue

            system_type, model_name, dataset_name = model_info 

            # Initialize new datasets
            if dataset_name not in model_results:
                model_results[dataset_name] = defaultdict(dict)

            # Skip all systems which don't match the criteria
            if system_type.lower() not in ('adaptive', fixed_type, 'randomized'):
                continue

            system_name = '{0} {1}'.format(system_type, model_name).upper()

            # Read the test log and get the accuracy for each budget matching the provided shift
            test_log = list(read_by_file_suffix(file_name))[0]
            noise_test_log = test_log[noise_key]

            for log_entry in noise_test_log.values():

                budget = log_entry['BUDGET']

                # Get the accuracy and power
                accuracy = log_entry['ACCURACY']
                power = log_entry['AVG_POWER']
                valid_accuracy = log_entry.get('VALID_ACCURACY')
                model_result = ModelResult(power=power, accuracy=accuracy, validation_accuracy=valid_accuracy)

                # Append accuracy to the adaptive model results
                if system_name not in model_results[dataset_name][budget]:
                    model_results[dataset_name][budget][system_name] = []

                model_results[dataset_name][budget][system_name].append(model_result)

    return model_results
