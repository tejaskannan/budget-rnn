import matplotlib.pyplot as plt
import numpy as np
from typing import List

from argparse import ArgumentParser
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dataset.data_manager import DataManager, get_data_manager
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID


def evaluate_features(features: np.ndarray, labels: np.ndarray, train_frac: float = 0.8) -> List[int]:
    """
    Evaluates the marginal impact of each feature in the given array (by retraining).

    Args:
        features: A [N, T, D] array of input features for each sequence element
        labels: A [N] array of labels per instance
    Returns:
        An (ordered) list of feature indices
    """
    # For feasibility purposes, we start with the first feature
    result: List[int] = [0]

    remaining_idx = list(range(1, features.shape[1]))

    split_point = int(features.shape[0] * train_frac)
    train_features = features[0:split_point, :, :]
    test_features = features[split_point:, :, :]

    train_labels = labels[0:split_point]
    test_labels = labels[split_point:]

    train_samples = train_features.shape[0]
    test_samples = test_features.shape[0]

    while len(remaining_idx) > 0:

        best_accuracy = 0.0
        best_idx = None

        for feature_idx in remaining_idx:
            feature_indices = result + [feature_idx]

            X_train = train_features[:, feature_indices, :].reshape(train_samples, -1)
            X_test = test_features[:, feature_indices, :].reshape(test_samples, -1)

            clf = LogisticRegression(max_iter=500)
            clf.fit(X_train, train_labels)
            accuracy = clf.score(X_test, test_labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_idx = feature_idx
    
        result.append(best_idx)
        remaining_idx.pop(remaining_idx.index(best_idx))

        print(best_accuracy)
        print(result)

    return result



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    args = parser.parse_args()

    # Load the input data
    data_manager = get_data_manager(folder=args.input_folder,
                                    sample_id_name=SAMPLE_ID,
                                    fields=[INPUTS, OUTPUT],
                                    extension='.jsonl.gz')
    data_manager.load()

    features: List[np.ndarray] = []
    labels: List[int] = []
    for i, sample in enumerate(data_manager.iterate(batch_size=64, should_shuffle=True)):
        label = sample[OUTPUT]

        labels.append(label)
        features.append(np.array(sample[INPUTS]))

    features = np.array(features)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.reshape(features.shape[0], -1))
    scaled_features = scaled_features.reshape(features.shape)

    ranked_features = evaluate_features(features=scaled_features, labels=np.array(labels))

    print(ranked_features)
