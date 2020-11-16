"""
Task of determining whether the sum of a sequence of numbers is the result of a linear
curve or cubic curve based on uniformly chosen points on [-5, 5]. We treat this as a unit test
for sequence models.
"""
import numpy as np
import os.path
from argparse import ArgumentParser

from utils.data_writer import DataWriter
from utils.file_utils import make_dir
from utils.constants import INPUTS, OUTPUT, TRAIN, VALID, TEST, SAMPLE_ID


MIN_VALUE = -5
MAX_VALUE = 5

TRAIN_FRAC = 0.7
VALID_FRAC = 0.15
TEST_FRAC = 0.15


def create_dataset(output_folder: str, num_samples: int, seq_length: int):
    with DataWriter(output_folder, file_prefix='data', file_suffix='jsonl.gz', chunk_size=5000) as writer:

        for sample_id in range(num_samples):
            # Randomly generate input points
            xs = np.sort(np.random.uniform(low=MIN_VALUE, high=MAX_VALUE, size=(seq_length, )))

            label = 1 if np.random.uniform(low=0.0, high=1.0) < 0.5 else 0

            if label == 1:  # Cubic
                inputs = np.power(xs, 3)
            else:  # Linear
                inputs = xs

            sample = {
                INPUTS: np.expand_dims(inputs, axis=-1).astype(float).tolist(),
                OUTPUT: label,
                SAMPLE_ID: sample_id
            }
            writer.add(sample)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--seq-length', type=int, default=5)
    args = parser.parse_args()

    # Create the output folder
    make_dir(args.output_folder)

    # Set the seed to make results reproducible
    np.random.seed(42)

    # Generate the (random) data for each partition
    partitions = [TRAIN, VALID, TEST]
    partition_fracs = [TRAIN_FRAC, VALID_FRAC, TEST_FRAC]

    for partition, frac in zip(partitions, partition_fracs):
        print('Starting Partition: {0}'.format(partition))

        create_dataset(output_folder=os.path.join(args.output_folder, partition),
                       num_samples=int(frac * args.num_samples),
                       seq_length=args.seq_length)
