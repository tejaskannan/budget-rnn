import serial
import time

from argparse import ArgumentParser
from typing import Optional, Iterable, List


def read_input_data(data_path: str, num_features: int, precision: int) -> Iterable[List[str]]:
    with open(data_path, 'r') as data_file:
        for line in data_file:
            tokens = line.split()

            features = [tokens[i:i+num_features] for i in range(0, len(tokens), num_features)]
            
            fixed_point_features: List[str] = []
            for feature_vector in features:
                # Convert the feature vector to floating point values
                fp_vec = [int(float(x) * (1 << precision)) for x in feature_vector]

                # Convert features to a space-separated string and end with a newline
                fixed_point_features.append(' '.join(map(str, fp_vec)) + '\n')
            
            yield fixed_point_features


def read_output_data(data_path: str) -> Iterable[int]:
    with open(data_path, 'r') as data_file:
        for line in data_file:
            yield int(line.strip())


def execute_client(input_data_path: str, output_data_path: str, port: str, num_features: int, precision: int, num_sequences: int, num_levels: int, delay: float, max_num_samples: Optional[int]):

    # Initialize client. Baudrate aligns with that of the bluetooth device.
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port = port
    ser.open()

    try:
        num_correct = 0
        for index, (features, output) in enumerate(zip(read_input_data(input_data_path, num_features, precision), read_output_data(output_data_path))):

            # Send all features
            num_written = 0
            for feature_index, feature_vec in enumerate(features):
                if feature_index % num_levels < num_sequences:
                    ser.write(feature_vec.encode('ascii'))
                    num_written += 1
                    print('Wrote {0} features.'.format(num_written), end='\r')

                # Always sleep to simulate the same sampling rate regardless of the number
                # of captured sequences
                time.sleep(delay)

            result = int(ser.read(1).decode())
            num_correct += 1 if result == output else 0

            print('\nResult: {0}. Expected: {1}'.format(result, output))
            print('========')

            if max_num_samples is not None and (index + 1) >= max_num_samples:
                break
    finally:
        ser.close()


if __name__ == '__main__':
    parser = ArgumentParser('Script to interface with devices running sequence models.')
    parser.add_argument('--input-data-file', type=str, required=True)
    parser.add_argument('--output-data-file', type=str, required=True)
    parser.add_argument('--port', type=str, required=True)
    parser.add_argument('--num-features', type=int, required=True)
    parser.add_argument('--precision', type=int, required=True)
    parser.add_argument('--num-sequences', type=int, required=True)
    parser.add_argument('--num-levels', type=int, required=True)
    parser.add_argument('--delay', type=float, default=1)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    assert args.delay > 0, 'Delay must be positive'
    assert args.max_num_samples is None or args.max_num_samples > 0, 'Max num samples must be positive'
    
    execute_client(input_data_path=args.input_data_file,
                   output_data_path=args.output_data_file,
                   port=args.port,
                   num_features=args.num_features,
                   precision=args.precision,
                   num_sequences=args.num_sequences,
                   num_levels=args.num_levels,
                   delay=args.delay,
                   max_num_samples=args.max_num_samples)
