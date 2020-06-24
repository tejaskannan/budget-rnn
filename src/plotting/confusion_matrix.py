import numpy as np
import matplotlib.pyplot as plt
import os.path
from argparse import ArgumentParser
from typing import Optional, Tuple, Dict, Any

from dataset.dataset_factory import get_dataset
from dataset.dataset import Dataset, DataSeries
from models.base_model import Model
from models.adaptive_model import AdaptiveModel
from models.standard_model import StandardModel
from models.model_factory import get_model
from utils.rnn_utils import get_logits_name
from utils.file_utils import extract_model_name, read_by_file_suffix, save_by_file_suffix
from utils.constants import TRAIN, TEST_LOG_PATH, HYPERS_PATH, METADATA_PATH, PREDICTION, OUTPUT, SMALL_NUMBER
from utils.np_utils import min_max_normalize, round_to_precision
from utils.hyperparameters import HyperParameters
from utils.adaptive_inference import threshold_predictions


def make_dataset(model_name: str, save_folder: str, dataset_type: str, dataset_folder: Optional[str]) -> Dataset:
    metadata_file = os.path.join(save_folder, METADATA_PATH.format(model_name))
    metadata = read_by_file_suffix(metadata_file)

    # Infer the dataset
    if dataset_folder is None:
        dataset_folder = os.path.dirname(metadata['data_folders'][TRAIN.upper()])

    # Validate the dataset folder
    assert os.path.exists(dataset_folder), f'The dataset folder {dataset_folder} does not exist!'

    return get_dataset(dataset_type=dataset_type, data_folder=dataset_folder)


def make_model(model_name: str, hypers: HyperParameters, save_folder: str) -> Model:
    model = get_model(hypers, save_folder, is_train=False)
    model.restore(name=model_name, is_train=False, is_frozen=False)
    return model


def get_serialized_info(model_path: str, dataset_folder: Optional[str]) -> Tuple[Model, Dataset, Dict[str, Any], str]:
    save_folder, model_file = os.path.split(model_path)

    model_name = extract_model_name(model_file)
    assert model_name is not None, f'Could not extract name from file: {model_file}'

    # Extract hyperparameters
    hypers_path = os.path.join(save_folder, HYPERS_PATH.format(model_name))
    hypers = HyperParameters.create_from_file(hypers_path)

    dataset = make_dataset(model_name, save_folder, hypers.dataset_type, dataset_folder)
    model = make_model(model_name, hypers, save_folder)

    # Get test log
    test_log_path = os.path.join(save_folder, TEST_LOG_PATH.format(model_name))
    assert os.path.exists(test_log_path), f'Must perform model testing before post processing'
    test_log = list(read_by_file_suffix(test_log_path))[0]

    return model, dataset, test_log, model_name


def create_confusion_matrix(model: Model, dataset: Dataset, thresholds: Optional[np.ndarray], precision: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    num_classes = model.metadata['num_classes']
    confusion_mat = np.zeros(shape=(num_classes, num_classes))
    
    data_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                 batch_size=model.hypers.batch_size,
                                                 metadata=model.metadata,
                                                 should_shuffle=False)
    
    for batch in data_generator:
        feed_dict = model.batch_to_feed_dict(batch, is_train=False)

        if isinstance(model, AdaptiveModel):
            logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]
            logits = model.execute(feed_dict, logit_ops)
            logits_concat = np.concatenate([np.expand_dims(logits[op], axis=1) for op in logit_ops], axis=1)

            # Normalize logits and round to fixed point representation
            normalized_logits = min_max_normalize(logits_concat, axis=-1)
            normalized_logits = round_to_precision(normalized_logits, precision=precision)

            predictions, _ = threshold_predictions(normalized_logits, thresholds)
        elif isinstance(model, StandardModel):
            predictions = model.execute(feed_dict, PREDICTION)
            predictions = predictions[PREDICTION]
        else:
            raise ValueError('Unsupported model type.')

        labels = np.squeeze(batch[OUTPUT])
        for pred, label in zip(predictions, labels):
            confusion_mat[label][pred] += 1

    label_sums = np.sum(confusion_mat, axis=0, keepdims=True)
    normalized_confusion_mat = confusion_mat / (label_sums + SMALL_NUMBER)

    return confusion_mat, normalized_confusion_mat


def plot_confusion_matrix(confusion_mat: np.ndarray, output_file: str):
     with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(confusion_mat, cmap=plt.get_cmap('magma'))

        for (i, j), z in np.ndenumerate(confusion_mat):
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        ax.set_title('Confusion Matrix on Test Set')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('True')
        plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--optimized-test-log', type=str)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--dataset-folder', type=str)
    args = parser.parse_args()

    model, dataset, test_log, model_name = get_serialized_info(args.model_path, args.dataset_folder)

    assert not (isinstance(model, AdaptiveModel) and (args.optimized_test_log is None or args.precision is None)), 'Must provide an optimized test log and precision for adaptive models'

    # Create the output file
    plot_output_file = os.path.join(model.save_folder, 'model-confusion-matrix-test-{0}.pdf'.format(model_name))
    data_output_file = os.path.join(model.save_folder, 'model-confusion-matrix-test-{0}.jsonl.gz'.format(model_name))

    # Collect the optimized thresholds (if provided)
    thresholds = None
    if args.optimized_test_log is not None:
        opt_test_log = list(read_by_file_suffix(args.optimized_test_log))[0]
        thresholds = np.array(opt_test_log['THRESHOLDS'])

    confusion_mat, normalized_confusion_mat = create_confusion_matrix(model, dataset, thresholds, precision=args.precision)
    
    plot_confusion_matrix(normalized_confusion_mat, plot_output_file)
    save_by_file_suffix([dict(confusion_mat=confusion_mat.tolist())], data_output_file)
