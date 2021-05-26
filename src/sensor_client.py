import numpy as np
import time
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple
from typing import Optional, Iterable, List

from ble_manager import BLEManager, ResponseType
from utils.file_utils import save_by_file_suffix


# NOTE: CHANGE THIS ADDRESS WITH THAT OF YOUR BLUETOOTH MODULE
MAC_ADDRESS = '00:35:FF:13:A3:1E'
BLE_HANDLE = 18
HCI_DEVICE = 'hci1'
MARGIN = 2

DataFiles = namedtuple('DataFiles', ['inputs_path', 'labels_path'])


def read_input_data(data_path: str, seq_length: int) -> Iterable[List[str]]:
    with open(data_path, 'r') as data_file:
        for line in data_file:
            tokens = line.split()

            num_features = int(len(tokens) / seq_length)
            features = [tokens[i:i+num_features] for i in range(0, len(tokens), num_features)]

            fixed_point_features: List[str] = []
            for feature_vector in features:
                # Convert features to a space-separated string and end with a newline. The newline
                # triggers the device to wake up.
                fixed_point_features.append(' '.join(map(str, feature_vector)) + '\n')

            yield fixed_point_features


def read_output_data(data_path: str) -> Iterable[int]:
    with open(data_path, 'r') as data_file:
        for line in data_file:
            yield int(line.strip())


def should_transmit(sample_idx: int, to_execute: Optional[int], stride_length: Optional[int], num_levels: Optional[int], is_completed: bool, is_skip: bool) -> bool:
    """
    Determines whether the server should transmit the sample with the given index based on information
    from the sensor device.
    """
    if is_completed:
        return False

    if sample_idx == 0 or to_execute is None or stride_length is None or num_levels is None:
        return True

    # Skip RNNs us the to_execute field to signal which samples to send
    if is_skip:
        return sample_idx >= to_execute

    current_level = int(sample_idx / num_levels) if stride_length == 1 else int(sample_idx % num_levels)
    return current_level <= to_execute


def execute_client(sample_freq: float, seq_length: int, max_sequences: int, data_files: DataFiles, output_file: str, budget: float, is_skip: bool, start_idx: int):
    """
    Starts the device client. This function either sends data and expects the device to respond with predictions
    or assumes that the device performs sensing on its own.

    Args:
        sample_freq: Sampling Frequency (Hz) for data collection
        seq_length: Number of samples in each sequence
        max_sequences: Maximum number of sequences before terminating collection. This must be provided if
            no data files are given.
        data_files: Information about data files containing already-collected datasets.
        output_file: File path in which to save results
        budget: The avg power budget in mW
        is_skip: Whether the model is a Skip RNN. This changes the wireless protocol slightly.
        start_idx: The starting index of the dataset
    """
    # Initialize the device manager
    device_manager = BLEManager(mac_addr=MAC_ADDRESS, handle=BLE_HANDLE, hci_device=HCI_DEVICE)

    # Get the experimental parameters
    sample_period = 1.0 / sample_freq
    energy_budget = (budget * sample_period * seq_length * max_sequences) / 1000  # Total Energy Budget in Joules

    # Read the data
    input_data = read_input_data(data_path=data_files.inputs_path, seq_length=seq_length)
    labels = read_output_data(data_path=data_files.labels_path)

    # Lists to store experiment results
    is_correct: List[float] = []
    predictions: List[float] = []
    processed_samples: List[float] = []
    device_voltage: List[float] = []

    # Variables to track energy consumption
    prev_energy = None
    total_energy = 0

    # Reset the device before we start
    device_manager.start()
    device_manager.reset_device()
    device_manager.stop()

    print('==========')
    print('Starting experiment with a Budget of {0}J ({1}mW) over {2} sequences'.format(energy_budget, budget, max_sequences))
    print('==========')

    num_sequences = 0
    for seq_idx, (seq_inputs, label) in enumerate(zip(input_data, labels)):
        if seq_idx < start_idx:
            continue

        if num_sequences >= max_sequences:
            break

        stride_length = None
        num_levels = None
        to_execute = None
        is_completed = False
        num_samples = 0

        on_times: List[float] = []

        for sample_idx, features in enumerate(seq_inputs):

            # Track the time elapsed when in contact with the device
            elapsed = 0.0

            # Determine whether we should send these features to the device
            should_send = should_transmit(sample_idx=sample_idx,
                                          to_execute=to_execute,
                                          stride_length=stride_length,
                                          num_levels=num_levels,
                                          is_completed=is_completed,
                                          is_skip=is_skip)

            if should_send:

                print('Sending Sample: {0}'.format(sample_idx), end='\r')
                num_samples += 1

                # Connect to the device
                start = time.time()
                device_manager.start(timeout=sample_period * MARGIN)

                # Transmit features and get the response
                response = device_manager.query(value=features, is_first=(sample_idx == 0))

                # Close the connection to save energy
                device_manager.stop()

                # Collect information obtained from the device
                if response.response_type == ResponseType.CONTROL:
                    stride_length = response.value.stride
                    num_levels = response.value.num_levels
                    to_execute = response.value.to_execute
                    # print(response.value)
                else:
                    prediction = response.value.prediction

                    # Compute the device energy in volts
                    voltage = response.value.voltage / 1000

                    predictions.append(prediction)
                    is_correct.append(float(prediction == label))
                    device_voltage.append(voltage)
                    processed_samples.append(num_samples)

                    accuracy_so_far = np.average(is_correct)
                    print('\nCompleted Seq {0} ({1}). Prediction: {2}, Voltage: {3:.5f}, Accuracy so Far: {4:.4f}'.format(num_sequences, seq_idx, prediction, voltage, accuracy_so_far))
                    print('==========')

                    is_completed = True

                end = time.time()
                elapsed = end - start

                on_times.append(elapsed)

            time.sleep(max(sample_period - elapsed, 0.1))

        num_sequences += 1

    accuracy = np.average(is_correct)
    print('Completed. Accuracy: {0:.4f}'.format(accuracy))

    # Save the results
    results_dict = {
        'accuracy': accuracy,
        'energy_budget': energy_budget,
        'count': len(predictions),
        'predictions': predictions,
        'is_correct': is_correct,
        'voltage': device_voltage,
        'processed_samples': processed_samples,
        'start_idx': start_idx,
        'budget': budget,
        'sample_period': sample_period

    }
    save_by_file_suffix([results_dict], output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sample-freq', type=float, required=True, help='The sample frequency in seconds')
    parser.add_argument('--seq-length', type=int, required=True, help='The sequence length of this dataset')
    parser.add_argument('--max-sequences', type=int, required=True, help='The maximum number of sequences to test')
    parser.add_argument('--inputs-path', type=str, required=True, help='The path to the input data (txt) file')
    parser.add_argument('--labels-path', type=str, required=True, help='The path to the label (txt) file')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the output (jsonl.gz) file')
    parser.add_argument('--budget', type=float, required=True, help='The energy budget')
    parser.add_argument('--is-skip', action='store_true', help='Whether this is a Skip RNN')
    parser.add_argument('--start-index', type=int, required=True, help='The optional element to start inference at')
    args = parser.parse_args()

    if os.path.exists(args.output_file):
        print('The output file {0} exists. Do you want to overwrite it? [Y/N]'.format(args.output_file))
        d = input()
        if d.lower() not in ('y', 'yes'):
            sys.exit(0)

    data_files = DataFiles(inputs_path=args.inputs_path,
                           labels_path=args.labels_path)

    execute_client(sample_freq=args.sample_freq,
                   seq_length=args.seq_length,
                   max_sequences=args.max_sequences,
                   data_files=data_files,
                   output_file=args.output_file,
                   budget=args.budget,
                   start_idx=args.start_index,
                   is_skip=args.is_skip)
