from argparse import ArgumentParser
from collections import Counter, deque
from typing import Iterable, Dict, Any, Optional, List

from utils.data_writer import DataWriter
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID, TIMESTAMP


def get_data_generator(input_path: str) -> Iterable[Dict[str, Any]]:
    with open(input_path, 'r') as input_file:
        data_iterator = iter(input_file)
        next(data_iterator)  # Skip the headings

        eth_counter: Counter = Counter()
        co_counter: Counter = Counter()
        for line in data_iterator:
            tokens = line.split()

            timestamp = float(tokens[0])
            concentrations = [float(tokens[1]), float(tokens[2])]
            inputs = list(map(float, tokens[3:]))

            # There are 16 sensors, so we should have 16 unique readings
            if len(inputs) < 16:
                continue

            yield {
                TIMESTAMP: timestamp,
                INPUTS: inputs,
                OUTPUT: concentrations
            }


def list_equal(first: List[float], second: List[float]) -> bool:
    if len(first) != len(second):
        return False

    for i in range(len(first)):
        if first[i] != second[i]:
            return False

    return True


def has_equal_outputs(output_list: List[List[float]]) -> bool:
    first_elem = output_list[0]
    for i in range(1, len(output_list)):
        elem = output_list[i] 
        if first_elem[0] != elem[0] or first_elem[1] != elem[1]:
            return False

    return True


def tokenize(input_file: str, output_folder: str, window: int, stride: int, chunk_size: int, max_num_samples: Optional[int]):
    data_generator = get_data_generator(input_file)

    with DataWriter(output_folder, file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size) as writer:
        
        data_window = deque()
        stride_counter = stride

        sample_id = 0
        for sample in data_generator:
            # Add sample to the data window            
            if len(data_window) > 0 and not list_equal(data_window[-1][OUTPUT], sample[OUTPUT]):
                data_window = deque()

            data_window.append(sample) 

            # Keep the data window of the desired length
            while len(data_window) > window:
                data_window.popleft()

            stride_counter += 1

            # Write the data window to the output file
            if len(data_window) == window and stride_counter >= stride:
                output = data_window[-1][OUTPUT]
                timestamp = data_window[-1][TIMESTAMP]
                features = [element[INPUTS] for element in data_window]

                sample_dict = {
                    SAMPLE_ID: sample_id,
                    OUTPUT: output,
                    TIMESTAMP: timestamp,
                    INPUTS: features
                }

                writer.add(sample_dict)
                stride_counter = 0
                sample_id += 1

            if (sample_id + 1) % chunk_size == 0:
                print('Completed {0} samples'.format(sample_id + 1), end='\r')

            if max_num_samples is not None and sample_id > max_num_samples:
                break

        print()

    print('Completed tokenization. Total of {0} samples'.format(sample_id))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--chunk-size' , type=int, default=5000)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    tokenize(input_file=args.input_file,
             output_folder=args.output_folder,
             window=args.window,
             stride=args.stride,
             chunk_size=args.chunk_size,
             max_num_samples=args.max_num_samples)
