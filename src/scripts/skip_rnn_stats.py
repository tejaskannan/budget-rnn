"""
This script measures the run-length of Skip RNNs on various datasets and targets.
"""
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple
from itertools import groupby
from typing import List

from models.base_model import Model
from models.standard_model import SKIP_GATES
from dataset.dataset import Dataset, DataSeries
from utils.file_utils import iterate_files
from utils.loading_utils import restore_neural_network


SkipResult = namedtuple('SkipResult', ['avg_samples', 'std_samples', 'target'])


def get_skip_results(model: Model, dataset: Dataset) -> np.ndarray:

    batch_iterator = dataset.minibatch_generator(series=DataSeries.VALID,
                                                 batch_size=64,
                                                 metadata=model.metadata,
                                                 should_shuffle=False)
    skip_gates: List[np.ndarray] = []
    for batch_num, batch in enumerate(batch_iterator):
        feed_dict = model.batch_to_feed_dict(batch, is_train=False, epoch_num=0)
        model_results = model.execute(feed_dict, ops=[SKIP_GATES])

        skip_gates.append(model_results[SKIP_GATES])

    skip_gates = np.vstack(skip_gates)

    # Calculate the avg number of samples consumed
    samples = np.sum(skip_gates, axis=-1)
    avg_samples = np.average(samples)
    std_samples = np.std(samples)

    return SkipResult(avg_samples=avg_samples, std_samples=std_samples, target=model.hypers.model_params['target_updates'])


def plot_results(results: List[SkipResult]):
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()

        xs: List[float] = []
        ys: List[float] = []
        yerr: List[float] = []
        for result in results:
            xs.append(result.target)
            ys.append(result.avg_samples)
            yerr.append(result.std_samples)

        xs, ys, yerr = zip(*sorted(zip(xs, ys, yerr), key=lambda t: t[0]))

        print(xs)
        print(ys)

        ax.errorbar(xs, ys, yerr=yerr, marker='o', label='Observed')
        ax.plot(xs, xs, marker='o', label='Target')

        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-folder', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    args = parser.parse_args()

    results: List[SkipResult] = []
    for model_path in iterate_files(args.model_folder, pattern=r'model-SKIP_RNN-*'):
        model, dataset = restore_neural_network(model_path, dataset_folder=args.dataset_folder)

        results.append(get_skip_results(model, dataset))

    plot_results(results)
