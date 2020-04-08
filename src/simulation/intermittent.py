import os.path
import numpy as np
import json

from argparse import ArgumentParser
from sklearn.metrics import f1_score
from typing import Iterable, Dict, Any, List, Tuple

from dataset.dataset import Dataset, DataSeries
from layers.output_layers import OutputType
from models.base_model import Model
from models.adaptive_model import AdaptiveModel
from models.standard_model import StandardModel
from utils.testing_utils import ClassificationMetric
from utils.constants import MODEL, PREDICTION, LOGITS, OUTPUT
from simulation_utils import get_serialized_info


MEGA_FACTOR = 1e-6
MICRO_FACTOR = 1e-6
MILLI_FACTOR = 1e-3
F1_SCORE_TYPE = 'micro'


def create_data_generator(model: Model, dataset: Dataset) -> Iterable[Dict[str, Any]]:
    data_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                 batch_size=1,
                                                 metadata=model.metadata,
                                                 should_shuffle=False,
                                                 drop_incomplete_batches=True)
    return data_generator


def consumed_energy(params: Dict[str, float], flop: float) -> float:
    noise = np.random.normal(loc=0.0, scale=params['noise'])
    return params['current'] * MILLI_FACTOR * (flop / params['processor']) + noise


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
    available_energy = 0.5 * params['capacitor'] * (params['system_voltage'] * params['system_voltage'])

    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []

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



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-paths', type=str, nargs='+')
    parser.add_argument('--params-file', type=str, required=True)
    parser.add_argument('--data-folder', type=str)
    args = parser.parse_args()

    assert os.path.exists(args.params_file), f'The file {args.params_file} does not exist!'
    with open(args.params_file) as f:
        params = json.load(f)

    # Extract all models before starting
    model_info = []
    for path in args.model_paths:
        assert os.path.exists(path), f'The path {path} does not exist.'
        model_info.append(get_serialized_info(path, args.data_folder))

    processed_batches: List[int] = []
    scores: List[float] = []
    for model, dataset, test_log in model_info:
        num_batches, f1 = run_simulation(model, dataset, test_log, params)

        processed_batches.append(num_batches)
        scores.append(f1)


    print(scores)
    print(processed_batches)
