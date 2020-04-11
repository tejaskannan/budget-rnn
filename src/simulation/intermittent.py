import os.path
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from sklearn.metrics import f1_score
from collections import namedtuple
from typing import Iterable, Dict, Any, List, Tuple

from dataset.dataset import Dataset, DataSeries
from layers.output_layers import OutputType
from models.base_model import Model
from models.adaptive_model import AdaptiveModel
from models.standard_model import StandardModel
from utils.testing_utils import ClassificationMetric
from utils.constants import MODEL, PREDICTION, LOGITS, OUTPUT
from utils.file_utils import save_by_file_suffix, read_by_file_suffix, make_dir
from simulation_utils import get_serialized_info


MICRO_FACTOR = 1e-6
MILLI_FACTOR = 1e-3
F1_SCORE_TYPE = 'micro'


PredictionOutput = namedtuple('PredictionOutput', ['prediction', 'logits', 'flops', 'energy'])


def create_data_generator(model: Model, dataset: Dataset) -> Iterable[Dict[str, Any]]:
    data_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                 batch_size=1,
                                                 metadata=model.metadata,
                                                 should_shuffle=False,
                                                 drop_incomplete_batches=True)
    return data_generator


def consumed_energy(params: Dict[str, float], flop: float) -> float:
    noise = np.random.normal(loc=0.0, scale=params['noise'])
    return (params['current'] + noise) * MILLI_FACTOR * params['system_voltage'] * (flop / params['processor'])


def standard_model_inference(sample: Dict[str, Any], model: Model, test_log: Dict[str, Any], params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, float]:
    feed_dict = model.batch_to_feed_dict(sample, is_train=False)
    output_dict = model.execute(ops=[PREDICTION, LOGITS], feed_dict=feed_dict)

    flop = test_log[MODEL][ClassificationMetric.FLOPS.name]
    return output_dict[PREDICTION], output_dict[LOGITS], consumed_energy(params, flop)


def adaptive_model_inference(sample: Dict[str, Any], model: Model, test_log: Dict[str, Any], available_energy: float, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, float]:
    feed_dict = model.batch_to_feed_dict(sample, is_train=False)
    prediction_generator = model.anytime_generator(feed_dict, model.num_outputs)

    current_prediction, current_logits = None, None
    current_energy = None

    for prediction_name, (prediction, logits) in zip(model.prediction_ops, prediction_generator):
        flop = test_log[prediction_name][ClassificationMetric.FLOPS.name]
        energy = consumed_energy(params, flop)    

        if current_prediction is None:
            current_prediction = prediction
        
        if current_logits is None:
            current_logits = logits

        if current_energy is None:
            current_energy = energy

        if energy >= available_energy:
            break
      
        current_prediction = prediction
        current_logits = logits
        current_energy = energy

    return current_prediction, current_logits, current_energy


def inference(sample: Dict[str, Any], model: Model, test_log: Dict[str, Any], available_energy: float, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, float]:
    if isinstance(model, AdaptiveModel):
        return adaptive_model_inference(sample, model, test_log, available_energy, params)
    if isinstance(model, StandardModel):
        return standard_model_inference(sample, model, test_log, params)


def run_simulation(model: Model, dataset: Dataset, test_log: Dict[str, Any], params: Dict[str, float]) -> Tuple[float, float]:
    # Create the data generator
    data_generator = create_data_generator(model, dataset)

    processed_batches = 0
    available_energy = 0.5 * (params['capacitor']) * (params['system_voltage'] * params['system_voltage'])

    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    # print('Available: {0}'.format(available_energy))

    for step in range(params['num_steps']):
        try:
            batch = next(data_generator)
        except StopIteration:
            data_generator = create_data_generator(model, dataset)
            batch = next(data_generator)

        prediction, logits, consumed_energy = inference(batch, model, test_log, available_energy, params)

        # Only add to completed batches if we didn't over-exhaust the energy
        if consumed_energy <= available_energy:
            processed_batches += 1

            predictions.append(np.vstack(prediction))
            labels.append(np.vstack(batch[OUTPUT]))

    # Compute the F1 score of processed samples
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)

    if model.output_type == OutputType.BINARY_CLASSIFICATION:
        f1 = f1_score(labels, predictions, average='binary')
    elif model.output_type == OutputType.MULTI_CLASSIFICATION:
        f1 = f1_score(labels, predictions, average=F1_SCORE_TYPE)

    return processed_batches, f1


def plot_results(result_dict: Dict[str, Dict[str, float]], output_file: str):
    with plt.style.context('fast'):
        fig, ax = plt.subplots()

        for label, results in sorted(result_dict.items()):
            score, success = results['f1_score'], results['success_frac']

            ax.scatter(score, success, label=label)
            
        ax.legend()
        ax.set_xlabel('F1 Score')
        ax.set_ylabel('Success Fraction')
        ax.set_title('Model Error versus Inference Success')

        plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-paths', type=str, nargs='+')
    parser.add_argument('--params-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--data-folder', type=str)
    args = parser.parse_args()

    # Fetch the parameters
    params = read_by_file_suffix(args.params_file)

    # Extract all models before starting
    model_info = []
    for path in args.model_paths:
        assert os.path.exists(path), f'The path {path} does not exist.'
        model_info.append(get_serialized_info(path, args.data_folder))

    processed_batches: List[int] = []
    scores: List[float] = []
    for i, (model, dataset, test_log) in enumerate(model_info):
        num_batches, f1 = run_simulation(model, dataset, test_log, params)

        processed_batches.append(num_batches)
        scores.append(f1)

        print(f'Completed model {i+1}/{len(model_info)}', end='\r')
    print()

    # Save results and generate plots
    result_dict: Dict[str, Dict[str, float]] = dict()
    for i in range(len(model_info)):
        score = scores[i]
        num_batches = processed_batches[i]
        model, _, _ = model_info[i]

        frac = num_batches / params['num_steps']
        result_dict[model.name] = dict(f1_score=score, num_batches=num_batches, success_frac=frac)

    make_dir(args.output_folder)
    results_file = os.path.join(args.output_folder, 'results.json')
    save_by_file_suffix(result_dict, results_file)

    plot_file = os.path.join(args.output_folder, 'plots.png')
    plot_results(result_dict, output_file=plot_file)
