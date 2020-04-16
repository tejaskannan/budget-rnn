from collections import namedtuple
from typing import List, Dict, Any, Iterable

from models.adaptive_model import AdaptiveModel
from dataset.rnn_sample_dataset import RNNSampleDataset
from dataset.dataset import DataSeries
from utils.threshold_utils import InferenceMode


OptimizerOutput = namedtuple('OptimizerOutput', ['thresholds', 'score'])


class ThresholdOptimizer:

    def __init__(self, iterations: int, batch_size: int, mode: str):
        self._batch_size = batch_size
        self._iterations = iterations
        self._mode = InferenceMode[mode.upper()]

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def inference_mode(self) -> InferenceMode:
        return self._mode

    def optimize(self, model: AdaptiveModel, dataset: RNNSampleDataset) -> OptimizerOutput:
        """
        Performs the threshold optimization on given model and dataset.
        """
        raise NotImplementedError()

    def get_data_generator(self, dataset: RNNSampleDataset, metadata: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        return dataset.minibatch_generator(DataSeries.TEST,
                                           batch_size=self.batch_size,
                                           metadata=metadata,
                                           should_shuffle=True,
                                           drop_incomplete_batches=False)


