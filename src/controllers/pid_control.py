import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import Tuple, List

from models.adaptive_model import AdaptiveModel
from dataset.dataset import DataSeries
from utils.rnn_utils import get_logits_name
from utils.file_utils import extract_model_name, read_by_file_suffix, save_by_file_suffix
from utils.np_utils import min_max_normalize, round_to_precision
from utils.constants import OUTPUT
from threshold_optimization.optimize_thresholds import get_serialized_info
from threshold_optimization.genetic_optimizer import threshold_predictions


class PIController:

    def __init__(self, kp: float, ki: float, output_range: Tuple[int, int]):
        self._kp = kp
        self._ki = ki

        self._output_range = output_range
        self._errors: List[float] = []
        self._times: List[float] = []
        self._integral = 0.0

    def errors(self) -> List[float]:
        return self._errors

    def times(self) -> List[float]:
        return self._times

    def step(self, y_true: int, y_pred: int, time: float) -> int:
        """
        Updates the controller and outputs the next control signal.
        """
        error = float(y_true - y_pred)

        if len(self._errors) > 0:
            h = (error + self._errors[-1]) / 2
            w = time - self._times[-1]
            self._integral += h * w

        integral_error = self._ki * self._integral
        proportional_error = self._kp * error

        control_signal = proportional_error + integral_error
        control_value = int(round(control_signal)) + y_pred

        self._errors.append(error)
        self._times.append(time)

        if control_value > self._output_range[1]:
            return self._output_range[1]
        elif control_value < self._output_range[0]:
            return self._output_range[0]
        else:
            return control_value

    def reset(self):
        """
        Resets the PI Controller.
        """
        self._errors = []
        self._times = []
        self._integral = 0.0


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--precision', type=int, required=True)
    parser.add_argument('--optimized-test-log', type=str)
    args = parser.parse_args()

    # Create the adaptive model
    model, dataset, test_log = get_serialized_info(args.model_path, dataset_folder=None)
    
    if args.optimized_test_log is None:
        thresholds = np.full(shape=(model.num_outputs, ), fill_value=1.0)
    else:
        opt_test_log = list(read_by_file_suffix(args.optimized_test_log))[0]
        thresholds = np.array(opt_test_log['THRESHOLDS'])

    # Execute model on the validation set and collect levels
    levels: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    accuracy: List[np.ndarray] = []

    logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]
    data_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                 batch_size=model.hypers.batch_size,
                                                 metadata=model.metadata,
                                                 should_shuffle=False)
    for batch in data_generator:
        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False)
        logits = model.execute(feed_dict, logit_ops)

        # Concatenate logits into a [B, L, C] array (logit_ops is already ordered by level).
        # For reference, L is the number of levels and C is the number of classes
        logits_concat = np.concatenate([np.expand_dims(logits[op], axis=1) for op in logit_ops], axis=1)

        # Normalize logits and round to fixed point representation
        normalized_logits = min_max_normalize(logits_concat, axis=-1)
        normalized_logits = round_to_precision(normalized_logits, precision=args.precision)

        pred, level = threshold_predictions(normalized_logits, thresholds)
        levels.append(level)

        true_values = np.squeeze(batch[OUTPUT])
        labels.append(true_values)
        accuracy.append((true_values == pred).astype(float))

    levels = np.concatenate(levels, axis=0)
    accuracy = np.concatenate(accuracy, axis=0)
    labels = np.concatenate(labels, axis=0)

    controller = PIController(kp=0.25, ki=0.0625, output_range=[1, model.num_outputs])
    y_pred = int(model.num_outputs / 2)  # Initialize to the median
    y_true = levels + 1

    max_time = len(y_true)

    predictions: List[int] = []
    errors: List[float] = []
    for t in range(max_time):
        predictions.append(y_pred)
        errors.append(y_true[t] - y_pred)

        # In a realistic situation, we don't know the number of levels if we did not collect enough data.
        # Thus, we use the average gap as the 'true' label to smooth out errors.
        target = y_true[t]
        if y_true[t] - y_pred > 0:
            target = int((model.num_outputs - y_pred) / 2) + y_pred

        y_pred = controller.step(y_true=target, y_pred=y_pred, time=t)

    with plt.style.context('ggplot'):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)

        times = list(range(max_time))
        ax1.plot(times, predictions, label='prediction')
        ax1.plot(times, y_true, label='true')
        ax1.legend()
        ax1.set_title('Predictions and True Signal')

        ax2.plot(times, errors, label='error')
        ax2.legend()
        ax2.set_title('Controller Error')

        cumulative_accuracy = np.cumsum(accuracy) / (np.array(times) + 1)
        ax3.plot(times, cumulative_accuracy, label='accuracy')
        ax3.legend()
        ax3.set_title('Model Accuracy over Time')

        ax4.plot(times, labels, label='true labels')
        ax4.legend()
        ax4.set_title('True Labels over Time')
        ax4.set_ylabel('Label Number')
        ax4.set_xlabel('Time')

        plt.tight_layout()
        plt.show()
