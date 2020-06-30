import numpy as np
import matplotlib.pyplot as plt
import math
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing import Tuple, List, Union

from models.adaptive_model import AdaptiveModel
from dataset.dataset import DataSeries
from utils.rnn_utils import get_logits_name
from utils.file_utils import extract_model_name, read_by_file_suffix, save_by_file_suffix
from utils.np_utils import min_max_normalize, round_to_precision
from utils.constants import OUTPUT, SMALL_NUMBER, INPUTS
from utils.adaptive_inference import normalize_logits, threshold_predictions
from threshold_optimization.optimize_thresholds import get_serialized_info
from level_bandit import LinearUCB
from logistic_regression_controller import Controller


POWER = np.array([24.085, 32.776, 37.897, 43.952, 48.833, 50.489, 54.710, 57.692, 59.212, 59.251])


def clip(x: int, bounds: Tuple[int, int]) -> int:
    if x > bounds[1]:
        return bounds[1]
    elif x < bounds[0]:
        return bounds[0]
    return x


class PIController:

    def __init__(self, kp: float, ki: float):
        self._kp = kp
        self._ki = ki

        self._errors: List[float] = []
        self._times: List[float] = []
        self._integral = 0.0

    def errors(self) -> List[float]:
        return self._errors

    def times(self) -> List[float]:
        return self._times

    def plant_function(self, y_pred: Union[float, int], proportional_error: float, integral_error: float) -> Union[float, int]:
        raise NotImplementedError()

    def step(self, y_true: Union[float, int], y_pred: Union[float, int], time: float) -> Union[float, int]:
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

        self._errors.append(error)
        self._times.append(time)

        return self.plant_function(y_pred, proportional_error, integral_error)

    def reset(self):
        """
        Resets the PI Controller.
        """
        self._errors = []
        self._times = []
        self._integral = 0.0


class ModelController(PIController):

    def __init__(self, kp: float, ki: float, output_range: Tuple[int, int]):
        super().__init__(kp, ki)
        self._output_range = output_range

    def plant_function(self, y_pred: Union[float, int], proportional_error: float, integral_error: float) -> Union[float, int]:
        control_signal = proportional_error + integral_error
        control_value = int(round(control_signal)) + y_pred

        if control_value > self._output_range[1]:
            control_value = self._output_range[1]
        elif control_value < self._output_range[0]:
            control_value = self._output_range[0]

        return control_value


class RandomController(PIController):
    
    def __init__(self, budget: float):
        super().__init__(0.0, 0.0)
        self._budget = budget

        power_array = np.array(POWER)
        weights = np.linalg.lstsq(power_array.reshape(1, -1), np.array([self._budget]))[0]

        self._weights = weights / np.sum(weights)
        self._indices = np.arange(start=0, stop=len(POWER))
        np.random.seed(42)  # For reproducible results
    
    def step(self, y_true: Union[float, int], y_pred: Union[float, int], time: float) -> Union[float, int]:
        return int(np.random.choice(self._indices, size=1, p=self._weights))


class BudgetController(PIController):

    def __init__(self, kp: float, ki: float, output_range: Tuple[int, int], budget: float, margin: float, power_factor: float):
        super().__init__(kp, ki)
        self._output_range = output_range
        self._budget = budget
        self._margin = margin
        self._power_factor = power_factor

        self._upper_limit = budget * (1 + margin)
        self._lower_limit = budget * (1 - margin)

    def plant_function(self, y_pred: Union[float, int], proportional_error: float, integral_error: float) -> Union[float, int]:
        # Error in power budget. For now, we only care about the upper bound budget
        control_signal = proportional_error + integral_error

        power = y_pred
        step = abs(control_signal) * self._power_factor

        if power >= self._lower_limit and power <= self._upper_limit:
            return 0  # By returning the highest # of levels, we allow the model controller to control
        
        sign = 1
        if power > self._upper_limit:
            sign = -1

        step = int(math.floor(sign * step))

        if step == 0:
            return sign
        return step


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--precision', type=int, required=True)
    parser.add_argument('--optimized-test-log', type=str)
    parser.add_argument('--dataset-folder', type=str)
    parser.add_argument('--budget', type=float)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--noise', type=float, default=1.0)
    parser.add_argument('--margin', type=float, default=0.01)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--controller', type=str, choices=['random', 'pi'], default='pi')
    args = parser.parse_args()

    # Create the adaptive model
    model, dataset, test_log = get_serialized_info(args.model_path, dataset_folder=args.dataset_folder)
    
    if args.optimized_test_log is None:
        thresholds = np.full(shape=(model.num_outputs, ), fill_value=1.0)
    else:
        opt_test_log = list(read_by_file_suffix(args.optimized_test_log))[0]
        thresholds = np.array(opt_test_log['THRESHOLDS'])

    # Using the budget, find the index of the best 'fixed' policy
    fixed_index = 0
    while fixed_index < model.num_outputs and POWER[fixed_index] < args.budget:
        fixed_index += 1

    fixed_index = fixed_index - 1
    # print(fixed_index)
    # print(POWER[fixed_index])

    # Execute model on the validation set and collect levels
    levels: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    power: List[np.ndarray] = []
    level_predictions: List[np.ndarray] = []
    context: List[np.ndarray] = []

    logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]
    data_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                 batch_size=model.hypers.batch_size,
                                                 metadata=model.metadata,
                                                 should_shuffle=args.shuffle)
    for batch_num, batch in enumerate(data_generator):
        inputs = np.array(batch[INPUTS])
        concat_input = inputs.reshape(inputs.shape[0], inputs.shape[2], -1)
        context.append(np.average(concat_input, axis=1))

        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False)
        logits = model.execute(feed_dict, logit_ops)

        # Concatenate logits into a [B, L, C] array (logit_ops is already ordered by level).
        # For reference, L is the number of levels and C is the number of classes
        logits_concat = np.concatenate([np.expand_dims(logits[op], axis=1) for op in logit_ops], axis=1)

        # Compute the predictions for each level
        level_pred = np.argmax(logits_concat, axis=-1)  # [B, L]
        level_predictions.append(level_pred)
        
        # Normalize logits and round to fixed point representation
        normalized_logits = normalize_logits(logits_concat, precision=args.precision)

        pred, level = threshold_predictions(normalized_logits, thresholds)
        levels.append(level)

        true_values = np.squeeze(batch[OUTPUT])
        labels.append(true_values)

    levels = np.concatenate(levels, axis=0)
    labels = np.concatenate(labels, axis=0)
    level_predictions = np.concatenate(level_predictions, axis=0)

    level_accuracy = np.average((level_predictions == np.expand_dims(labels, axis=1)).astype(float), axis=0)

    output_range = (0, model.num_outputs - 1)

#    if args.controller == 'random':
#        controller = RandomController(budget=args.budget)
#    else:
#        controller = ModelController(kp=1.0, ki=1.0/32.0, output_range=output_range)

    # controller = LinearUCB(num_features=context.shape[1], num_arms=model.num_outputs, alpha=1.0)
    
    # Fit controller and make predictions on test set. We would usually compute the predictions
    # incrementally, but we do them all at once for efficiency purposes.
    controller = Controller(model=model, dataset=dataset, precision=args.precision, budgets=[args.budget], share_model=True, trials=5)
    controller.fit(series=DataSeries.VALID)
    predicted_levels = controller.predict_levels(series=DataSeries.TEST, budget=args.budget)
    print(np.average(predicted_levels))

    print('Controller Train Acc: {0:.5f}'.format(controller.score(series=DataSeries.VALID)))
    print('Controller Test Acc: {0:.5f}'.format(controller.score(series=DataSeries.TEST)))

    budget_controller = BudgetController(kp=1.0, ki=0.0625, output_range=output_range, budget=args.budget, margin=args.margin, power_factor=1.0)

    y_pred = int(model.num_outputs / 2)  # Initialize to the median
    y_pred_model = y_pred
    y_true = predicted_levels

    max_time = len(y_true)
    times = list(range(max_time))

    power: List[float] = []
    predictions: List[int] = []
    errors: List[float] = []
    num_correct: List[float] = []
    fixed_correct: List[float] = []
    controller_accuracy: List[float] = []

   # max_level = 0
   # for level, t in enumerate(thresholds):
   #     if t < 1e-7:
   #         break
   #     max_level = level

    for t in range(max_time):
        #if t == 0:
        #    y_pred_model = controller.predict(x=np.zeros_like(context[0]))
        #else:
        #    y_pred_model = controller.predict(x=context[t-1])

        y_pred_model = predicted_levels[t]

        # Make adjustments based on observed power
        avg_power = args.budget
        if len(power) > 0:
            avg_power = np.clip(np.average(power), a_min=args.budget, a_max=None)
        budget_step = budget_controller.step(y_true=args.budget, y_pred=avg_power, time=t)

        y_pred = clip(y_pred_model + budget_step, bounds=output_range)

        # Compute the (noisy) power consumption for using this number of levels
        p = POWER[y_pred] + np.random.uniform(low=-args.noise, high=args.noise)

        power.append(p)
        predictions.append(y_pred)
        errors.append(y_pred_model - y_pred)
        controller_accuracy.append(float(y_pred == y_pred_model))

        # In a realistic situation, we don't know the number of levels if we did not collect enough data.
        # Thus, we use the average gap as the 'true' label to smooth out errors.
      #  target = y_true[t]
      #  if y_true[t] - y_pred > SMALL_NUMBER:
      #      target = int((model.num_outputs - y_pred) / 2) + y_pred
      #      model_prediction = level_predictions[t, y_pred]  # Get the prediction of the maximum level if we under-shot the prediction
      #  else:
      #      model_prediction = level_predictions[t, y_true[t]]  # Get the prediction from the stopped level

        model_prediction = level_predictions[t, y_pred]

        # Compute the average power after adding in the new sample
        #avg_power = np.clip(np.average(power), a_min=args.budget, a_max=None)

        ## Update the controller
        #y_pred_model = controller.step(y_true=target, y_pred=y_pred_model, time=t)
        #budget_step = budget_controller.step(y_true=args.budget, y_pred=avg_power, time=t)
        #y_pred = clip(y_pred_model + budget_step, bounds=output_range)

        num_correct.append(float(model_prediction == labels[t]))

        # Record the prediction of the fixed policy
        fixed_prediction = level_predictions[t, fixed_index]
        fixed_correct.append(float(fixed_prediction == labels[t]))

        # Structure reward
        #bandit_reward = target - y_pred_model if target < y_pred_model else 0.1 * (y_pred_model - target)
       # bandit_target = max_level if y_true[t] > y_pred else y_true[t]
       # reward_factor = 1.0 if y_pred_model > target else 1.0
       # bandit_reward = -1.0 * np.abs(bandit_target - y_pred_model)

       # controller.update(arm=y_pred_model, reward=bandit_reward, x=context[t])

       # if avg_power > (args.budget * (1 + args.margin)):
       #     print('Power: {0}, Y_pred: {1}, Budget Step: {2}'.format(avg_power, y_pred, budget_step))

    print('Controller Accuracy: {0}'.format(np.average(controller_accuracy)))

    avg_power = np.cumsum(power) / (np.array(times) + 1)

    with plt.style.context('ggplot'):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(figsize=(16, 12), nrows=5, ncols=1, sharex=True)

        ax1.plot(times, y_true, label='true')
        ax1.legend()
        ax1.set_title('True Model Levels over Time')

        ax2.plot(times, errors, label='error')
        ax2.legend()
        ax2.set_title('Model Controller Error')

        cumulative_accuracy = np.cumsum(num_correct) / (np.array(times) + 1)
        cumulative_fixed_accuracy = np.cumsum(fixed_correct) / (np.array(times) + 1)

        print('Accuracy: Adaptive -> {0:.4f}, Fixed -> {1:.4f}'.format(cumulative_accuracy[-1], cumulative_fixed_accuracy[-1]))
        ax3.plot(times, cumulative_accuracy, label='Adaptive')
        ax3.plot(times, cumulative_fixed_accuracy, label='Fixed')
        ax3.legend()
        ax3.set_title('Model Accuracy over Time')

        ax4.plot(times, labels, label='true labels')
        ax4.legend()
        ax4.set_title('True Labels over Time')
        ax4.set_ylabel('Label Number')

        print('Adaptive Power: {0:.4f}'.format(avg_power[-1]))
        power_budget = [args.budget for _ in times]
        ax5.plot(times, avg_power, label='Avg Power')
        ax5.plot(times, power_budget, label='Budget')
        ax5.legend()
        ax5.set_title('Cumulative Average Power')
        ax5.set_xlabel('Time')

        plt.tight_layout()

        if args.output_file is not None:
            plt.savefig(args.output_file)
        else:
            plt.show()
