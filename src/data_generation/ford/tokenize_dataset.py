import re
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter
from scipy.signal import spectrogram
from typing import Iterable, Dict, Any, Optional

from utils.data_writer import DataWriter
from utils.constants import SAMPLE_ID, INPUTS, OUTPUT, SMALL_NUMBER


STRIDE = 50



def dataset_iterator(input_file: str, window_size: int, noise: float, reps: int, num_features: Optional[int]) -> Iterable[Dict[str, Any]]:

    with open(input_file, 'r') as fin:

        is_header = True

        sample_id = 0
        for line in fin:
            if line.strip() == '@data':
                is_header = False
            elif not is_header:
                tokens = [float(t.strip()) for t in re.split(r'[,:]+', line)]

                input_data = np.array(tokens[:-1])
                label = int((tokens[-1] + 1) / 2)  # Map labels to 0 / 1

                input_values = np.copy(input_data)
                for i in range(reps):

                    if abs(noise) > SMALL_NUMBER and i > 0:
                        input_noise = np.random.uniform(low=-noise, high=noise, size=input_data.shape)
                        input_values = input_data + input_noise

                    # Compute the signal spectrogram
                    f, t, Sxx = spectrogram(input_values, nperseg=window_size, scaling='spectrum', noverlap=0)

                    if num_features is not None:
                        Sxx = Sxx[0:num_features, :]

                    #plt.pcolormesh(t, f[0:num_features], Sxx)
                    #plt.colorbar()
                    #plt.show()

                    # Split the spectrogram into features
                    features = [Sxx[:, i].astype(float).tolist() for i in range(Sxx.shape[1])]

                    yield {
                        SAMPLE_ID: sample_id,
                        INPUTS: features,
                        OUTPUT: label
                    }

                sample_id += 1


def tokenize_dataset(input_file: str, output_folder: str, window_size: int, noise: float, reps: int, num_features: Optional[int]):

    with DataWriter(output_folder, file_prefix='data', chunk_size=10000, file_suffix='jsonl.gz') as writer:
        
        label_counter: Counter = Counter()

        for index, sample in enumerate(dataset_iterator(input_file, window_size, noise=noise, reps=reps, num_features=num_features)):
            label_counter[sample[OUTPUT]] += 1
            writer.add(sample)

            if (index + 1) % 500 == 0:
                print('Completed {0} samples.'.format(index + 1), end='\r')

    print()
    print(label_counter)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--num-features', type=int)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--reps', type=int, default=1)
    args = parser.parse_args()

    tokenize_dataset(input_file=args.input_file,
                     output_folder=args.output_folder,
                     window_size=args.window_size,
                     reps=args.reps,
                     noise=args.noise,
                     num_features=args.num_features)
