import numpy as np
from typing import Dict

from models.adaptive_model import AdaptiveModel
from dataset.rnn_sample_dataset import RNNSampleDataset
from enum import Enum, auto
from utils.rnn_utils import get_logits_name
from utils.np_utils import sigmoid
from utils.constants import OUTPUT, BIG_NUMBER, SMALL_NUMBER

from .threshold_optimizer import ThresholdOptimizer, OptimizerOutput
from .optimization_utils import f1_loss, f1_loss_gradient


LOWER_BOUND = 0.05
UPPER_BOUND = 0.95


class UpdateTypes(Enum):
    SGD = auto()
    NESTEROV = auto()
    RMSPROP = auto()
    ADAM = auto()


class GradientUpdate:
    
    def __init__(self, learning_rate: float, beta: float, argmin_weight: float):
        self._learning_rate = learning_rate
        self._argmin_weight = argmin_weight
        self._beta = beta

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def argmin_weight(self) -> float:
        return self._argmin_weight

    def apply(self, probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, sharpen_factor: float, step: int) -> np.ndarray:
        raise NotImplementedError()


class GradientOptimizer(ThresholdOptimizer):

    def __init__(self,
                 iterations: int,
                 batch_size: int,
                 update_type: str,
                 update_params: Dict[str, float],
                 sharpen_factor: float,
                 tolerance: float,
                 beta: float,
                 argmin_weight: float):
        super().__init__(iterations, batch_size)
        self._sharpen_factor = sharpen_factor
        self._tolerance = tolerance
        self._beta = beta
        self._argmin_weight = argmin_weight

        self._update_type = UpdateTypes[update_type.upper()]
        if self._update_type == UpdateTypes.SGD:
            self._updater = SGDUpdate(learning_rate=update_params['learning_rate'], beta=beta, argmin_weight=argmin_weight)
        elif self._update_type == UpdateTypes.NESTEROV:
            self._updater = NesterovUpdate(learning_rate=update_params['learning_rate'],
                                           momentum=update_params['momentum'],
                                           beta=beta,
                                           argmin_weight=argmin_weight)
        elif self._update_type == UpdateTypes.RMSPROP:
            self._updater = RMSPropUpdate(learning_rate=update_params['learning_rate'],
                                          gamma=update_params['gamma'],
                                          beta=beta,
                                          argmin_weight=argmin_weight)
        elif self._update_type == UpdateTypes.ADAM:
            self._updater = AdamUpdate(learning_rate=update_params['learning_rate'],
                                       first_momentum=update_params['first_momentum'],
                                       second_momentum=update_params['second_momentum'],
                                       beta=beta,
                                       argmin_weight=argmin_weight)
        else:
            raise ValueError(f'Unknown update type: {self._update_type}.')

    @property
    def sharpen_factor(self) -> float:
        return self._sharpen_factor

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @property
    def update_type(self) -> UpdateTypes:
        return self._update_type

    @property
    def updater(self) -> GradientUpdate:
        return self._updater

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def argmin_weight(self) -> float:
        return self._argmin_weight

    def optimize(self, model: AdaptiveModel, dataset: RNNSampleDataset) -> OptimizerOutput:
        """
        Runs the gradient-based optimization of thresholds for the given model and dataset.
        """
        # Create the data iterator
        data_generator = self.get_data_generator(dataset, model.metadata)

        # Set logit operations
        logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]

        # Initialize thresholds to 0.5
        thresholds = np.full(shape=(model.num_outputs,), fill_value=0.5) 

        # Track loss
        loss = BIG_NUMBER

        batch = next(data_generator)
        for batch_num in range(self.iterations):
            feed_dict = model.batch_to_feed_dict(batch, is_train=False)
            logits = model.execute(feed_dict, logit_ops)

            # Concatenate logits into a 2D array (logit_ops is already ordered by level)
            logits_concat = np.concatenate([logits[op] for op in logit_ops], axis=-1)
            probabilities = sigmoid(logits_concat)
            labels = np.squeeze(np.vstack(batch[OUTPUT]), axis=-1)

            # Compute loss
            loss = f1_loss(probabilities, labels, thresholds, self.sharpen_factor, beta=self.beta, argmin_weight=self.argmin_weight)
            
            # Perform update
            next_thresholds = self.updater.apply(probabilities, labels, thresholds, self.sharpen_factor, batch_num + 1)

            # Check convergence
            should_stop = bool(np.linalg.norm(thresholds - next_thresholds, ord=2) < self.tolerance)
            thresholds = next_thresholds
            if should_stop:
                break

            try:
                batch = next(data_generator)
            except StopIteration:
                data_generator = self.get_data_generator(dataset, model.metadata)
                batch = next(data_generator)

            print(f'Completed {batch_num + 1} / {self.iterations} iterations. Loss so far: {loss:.3f}.', end='\r')

        print()

        return OptimizerOutput(score=-1 * loss, thresholds=list(thresholds))


class SGDUpdate(GradientUpdate):

    def apply(self, probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, sharpen_factor: float, step: int) -> np.ndarray:
        gradient = f1_loss_gradient(probabilities, labels, thresholds, sharpen_factor, beta=self.beta, argmin_weight=self.argmin_weight)
        return np.clip(thresholds - self.learning_rate * gradient, a_min=LOWER_BOUND, a_max=UPPER_BOUND)


class NesterovUpdate(GradientUpdate):

    def __init__(self, learning_rate: float, beta: float, argmin_weight: float, momentum: float):
        super().__init__(learning_rate, beta, argmin_weight)
        self._momentum = momentum
        self._momentum_vector = None

    @property
    def momentum(self) -> float:
        return self._momentum

    def apply(self, probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, sharpen_factor: float, step: int) -> np.ndarray:
        # Initialize momentum vector if needed
        if self._momentum_vector is None:
            self._momentum_vector = np.zeros_like(thresholds)

        diff = np.clip(thresholds - self._learning_rate * self._momentum_vector, a_min=LOWER_BOUND, a_max=UPPER_BOUND)
        gradient = f1_loss_gradient(probabilities, labels, diff, sharpen_factor, beta=self.beta, argmin_weight=self.argmin_weight)
        self._momentum_vector = self.learning_rate * self._momentum_vector + self.momentum * gradient

        return np.clip(thresholds - self._momentum_vector, a_min=LOWER_BOUND, a_max=UPPER_BOUND)


class RMSPropUpdate(GradientUpdate):

    def __init__(self, learning_rate: float, beta: float, argmin_weight: float, gamma: float):
        super().__init__(learning_rate, beta, argmin_weight)
        self._gamma = gamma
        self._expected_sq_grad = None

    @property
    def gamma(self) -> float:
        return self._gamma

    def apply(self, probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, sharpen_factor: float, step: int) -> np.ndarray:
        gradient = f1_loss_gradient(probabilities, labels, thresholds, sharpen_factor, beta=self.beta, argmin_weight=self.argmin_weight)

        if self._expected_sq_grad is None:
            self._expected_sq_grad = np.square(gradient)
        else:
            self._expected_sq_grad = (1.0 - self.gamma) * self._expected_sq_grad + self.gamma * np.square(gradient)

        weighted_learn_rate = self.learning_rate / (np.sqrt(self._expected_sq_grad) + SMALL_NUMBER)  # [L]

        return np.clip(thresholds - weighted_learn_rate * gradient, a_min=LOWER_BOUND, a_max=UPPER_BOUND)


class AdamUpdate(GradientUpdate):

    def __init__(self, learning_rate: float, beta: float, argmin_weight: float, first_momentum: float, second_momentum: float):
        super().__init__(learning_rate, beta, argmin_weight)

        self._first_momentum = first_momentum
        self._second_momentum = second_momentum

        self._first_moment = None
        self._second_moment = None

    @property
    def first_momentum(self) -> float:
        return self._first_momentum

    @property
    def second_momentum(self) -> float:
        return self._second_momentum

    def apply(self, probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, sharpen_factor: float, step: int) -> np.ndarray:
        # Initialize moments if necessary
        if self._first_moment is None:
            self._first_moment = np.zeros_like(thresholds)

        if self._second_moment is None:
            self._second_moment = np.zeros_like(thresholds)

        # Compute gradient
        gradient = f1_loss_gradient(probabilities, labels, thresholds, sharpen_factor, beta=self.beta, argmin_weight=self.argmin_weight)

        # Compute approximate moments
        self._first_moment = self._first_momentum * self._first_moment + (1.0 - self._first_momentum) * gradient
        self._second_moment = self._second_momentum * self._second_moment + (1.0 - self._second_momentum) * np.square(gradient)

        # Apply bias corrections
        first_moment = self._first_moment / (1.0 - np.power(self._first_momentum, step))
        second_moment = self._second_moment / (1.0 - np.power(self._second_momentum, step))

        weighted_learning_rate = self.learning_rate / (np.sqrt(second_moment) + SMALL_NUMBER)
        return np.clip(thresholds - weighted_learning_rate * first_moment, a_min=LOWER_BOUND, a_max=UPPER_BOUND)
