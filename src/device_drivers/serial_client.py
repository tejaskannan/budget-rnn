import serial
import time

from argparse import ArgumentParser
from typing import Optional, Iterable, List


def read_input_data(data_path: str, num_features: int) -> Iterable[List[str]]:
    with open(data_path, 'r') as data_file:
        for line in data_file:
            tokens = line.split()

            features = [tokens[i:i+num_features] for i in range(0, len(tokens), num_features)]
            
            fixed_point_features: List[str] = []
            for feature_vector in features:
                # Convert features to a space-separated string and end with a newline
                fixed_point_features.append(' '.join(map(str, feature_vector)) + '\n')
            
            yield fixed_point_features


def read_output_data(data_path: str) -> Iterable[int]:
    with open(data_path, 'r') as data_file:
        for line in data_file:
            yield int(line.strip())


def execute_client(input_data_path: str, output_data_path: str, port: str, num_features: int, num_sequences: int, num_levels: int, max_num_samples: Optional[int]):

    # Fetch the input and output data
    input_data = list(read_input_data(input_data_path, num_features))
    output_data = list(read_output_data(output_data_path))

    # Initialize client. Baudrate aligns with that of the bluetooth device.
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port = port
    ser.open()

    try:
        # Step 1: Send the start message
        ser.write('B'.encode('ascii'))

        # Features is a list of feature vectors for the current sequence, output is the label (not sent over the link)
        num_correct = 0
        seq_index = 0
        for index, (features, label) in enumerate(zip(input_data, output_data)):

            # Step 2: Receive a 'pull' data request from the device. This request comes in two forms.
            #   Option 1 -> A message of the type SX where X is the sequence index to send (encoded as an unsigned byte).
            #               The server responds with the corresponding data features
            #   Option 2 -> A message of the type PY where Y is the predicted class index. In this case, the server responds
            #               with the first element of the NEXT sequence. This behavior is implicit to limit communication,
            #               as all models always process the first sample
            should_stop = False
            while not should_stop:
                # Wait on a pull message
                pull_message = ser.read(2).decode()

                # Extract the message components
                command = pull_message[0]
                val = int(ord(pull_message[1]))

                print('Command: {0}, Val: {1}'.format(command, val))

                # Detect the message type and perform the corresponding action
                if command == 'P':
                    print('Predicted Class: {0}, True Label: {1}'.format(val, label))
                    should_stop = True
                    num_correct += int(val == label)
                else:
                    feature_vec = features[val]
                    print(feature_vec, end='')
                    ser.write(feature_vec.encode('ascii'))

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
    parser.add_argument('--num-sequences', type=int, required=True)
    parser.add_argument('--num-levels', type=int, required=True)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    assert args.max_num_samples is None or args.max_num_samples > 0, 'Max num samples must be positive'
    
    execute_client(input_data_path=args.input_data_file,
                   output_data_path=args.output_data_file,
                   port=args.port,
                   num_features=args.num_features,
                   num_sequences=args.num_sequences,
                   num_levels=args.num_levels,
                   max_num_samples=args.max_num_samples)
