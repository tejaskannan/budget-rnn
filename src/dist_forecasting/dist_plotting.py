import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os.path
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from sklearn.linear_model import LogisticRegression
from typing import Any, Dict, DefaultDict, List, Tuple, Optional

from models.standard_model import StandardModel
from models.model_factory import get_model
from dataset.dataset import Dataset, DataSeries
from dataset.dataset_factory import get_dataset
from utils.constants import INPUTS, OUTPUT, SEQ_LENGTH, DROPOUT_KEEP_RATE
from utils.constants import ACTIVATION_NOISE, LOGITS
from utils.np_utils import pad_array, softmax

from dist_utils import prob_fn, format_label, make_model


Stats = namedtuple('Stats', ['mean', 'std', 'accuracy'])


def subsample_batch(batch: Dict[str, Any], num_samples: int, length: int, rand: np.random.RandomState) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        INPUTS: [],
        OUTPUT: [],
        SEQ_LENGTH: [],
        DROPOUT_KEEP_RATE: 1.0,
        ACTIVATION_NOISE: 0.0001
    }
  
    seq_length = batch[INPUTS][0].shape[0]
    
    for _ in range(num_samples):
        # Generate the indices
        subseq_idx = rand.choice(seq_length, size=length + 1, replace=False)
        if 0 not in subseq_idx:
            subseq_idx[-1] = 0

        subseq_idx = np.sort(subseq_idx)

        # Generate the sub-sample
        inputs = batch[INPUTS][0][subseq_idx, :]
        inputs = pad_array(inputs, new_size=seq_length, value=0, axis=0)  # [T, D]
        result[INPUTS].append(inputs)

        result[OUTPUT].append(batch[OUTPUT][0])
        result[SEQ_LENGTH].append(length + 1)

    return result


def execute_standard_model(model: StandardModel, dataset: Dataset, series: DataSeries, trials: int, dist_mode: str) -> Tuple[Dict[int, Stats], List[Tuple[float, float]]]:
    """
    Executes the neural network on the given data series. We do this in a separate step
    to avoid recomputing for multiple budgets. Executing the neural network is relatively expensive.

    Args:
        model: The standard model used to perform inference
        dataset: The dataset to perform inference on
        series: The data series to extract. This is usually the TEST set.
    Returns:
        A model result tuple containing the inference results.
    """
    dist_results = defaultdict(list)
    correct_results = defaultdict(list)
    dist_dataset = []

    # Make the batch generator. Don't shuffle so we have consistent results.
    data_generator = dataset.minibatch_generator(series=series,
                                                 batch_size=1,
                                                 metadata=model.metadata,
                                                 should_shuffle=False)

    rand = np.random.RandomState(seed=421)

    for batch_num, batch in enumerate(data_generator):

        seq_length = batch[INPUTS][0].shape[0]

        for length in range(seq_length):
            subsampled_batch = subsample_batch(batch=batch, length=length, num_samples=trials, rand=rand)

            # Compute the predicted probabilities
            feed_dict = model.batch_to_feed_dict(subsampled_batch, is_train=False, epoch_num=0)

            model_results = model.execute(feed_dict, [LOGITS])  # [B, L]
            probs = softmax(model_results[LOGITS], axis=-1)  # [B, L]
            pred = np.argmax(probs, axis=-1)  # [B]

            is_correct = np.isclose(pred, batch[OUTPUT][0]).astype(float)
            dist_aggregate = prob_fn(dist=probs, name=dist_mode)

            # Add to the binary classification entropy data
            for i in range(trials):
                dist_dataset.append((dist_aggregate[i], is_correct[i]))

            # Compute the average entropy
            avg_dist = np.average(dist_aggregate)

            dist_results[length].append(avg_dist)
            correct_results[length].extend(is_correct)

    # Compute aggregate states
    stats: Dict[int, Stats] = dict()
    for length, avg_dist in dist_results.items():
        num_correct = correct_results[length]
        stats[length] = Stats(mean=np.average(avg_dist),
                              std=np.std(avg_dist),
                              accuracy=np.average(num_correct))

    return stats, dist_dataset


def plot(stats: Dict[int, Stats], dist_mode: str, clf_accuracy: float, output_file: Optional[str]):
    dist_mode = format_label(dist_mode)

    with plt.style.context('seaborn-ticks'):
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()

        accuracy: List[float] = []
        dist: List[float] = []
        xs: List[int] = []

        for length, length_stats in stats.items():
            dist.append(length_stats.mean)
            accuracy.append(length_stats.accuracy)
            xs.append(length + 1)

        ax1.plot(xs, dist, label='Avg {0}'.format(dist_mode), color='blue', marker='o')
        ax2.plot(xs, accuracy, label='Accuracy', color='red', marker='o')

        # Set labels and axis parameters
        ax1.set_xlabel('Seq Length')
        ax1.set_ylabel('Avg {0}'.format(dist_mode), color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2.set_ylabel('Accuracy', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax1.set_title('{0} and Accuracy'.format(dist_mode))

        # Write the classification accuracy
        y_lower, y_upper = ax1.get_ylim()
        y_half = (y_upper + y_lower) / 2
        ax1.text(len(stats) * 0.75, y_half, 'CLF Acc: {0:.4f}'.format(clf_accuracy), fontsize=12)
    
        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-file', type=str, required=True)
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--dist-mode', type=str, required=True, choices=['entropy', 'max_prob'])
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    model = make_model(model_path=args.model_file)
    dataset = get_dataset(dataset_type='randomized', data_folder=args.data_folder)

    results, dist_dataset = execute_standard_model(model=model,
                                                   dataset=dataset,
                                                   series=DataSeries.VALID,
                                                   trials=4,
                                                   dist_mode=args.dist_mode)

    for length, stats in results.items():
        print('{0} -> {1:.4f} ({2:.4f}), {3:.4f}'.format(length, stats[0], stats[1], stats[2]))

    # Fit the classifier
    dist_dataset = np.array(dist_dataset)
    X, y = dist_dataset[:, 0], dist_dataset[:, 1]
    X = X.reshape(-1, 1)

    clf = LogisticRegression(penalty='none').fit(X, y)
    clf_accuracy = clf.score(X, y)
    print('Classifier Score: {0:.4f}'.format(clf_accuracy))

    # Plot the results
    plot(stats=results, dist_mode=args.dist_mode, clf_accuracy=clf_accuracy, output_file=args.output_file)
