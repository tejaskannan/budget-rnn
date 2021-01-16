import numpy as np
import os.path
from collections import namedtuple
from typing import List

from models.model_factory import get_model
from models.standard_model import StandardModel
from utils.constants import HYPERS_PATH
from utils.file_utils import extract_model_name
from utils.hyperparameters import HyperParameters


RolloutResult = namedtuple('RolloutResult', ['states', 'rewards'])


def entropy(dist: np.ndarray) -> np.ndarray:
    dist_log = np.log(dist)
    return -1 * np.sum(dist * dist_log, axis=-1)


def max_prob(dist: np.ndarray) -> np.ndarray:
    return np.max(dist, axis=-1)


def prob_fn(dist: np.ndarray, name: str) -> np.ndarray:
    fn = entropy if name.lower() == 'entropy' else max_prob
    return fn(dist=dist)


def stack_rollouts(rollouts: List[RolloutResult]) -> RolloutResult:
    concat = RolloutResult(states=[], rewards=[])
    for rollout_values in rollouts:
        concat.states.extend(rollout_values.states)
        concat.rewards.extend(rollout_values.rewards)

    return RolloutResult(states=np.vstack(concat.states),
                         rewards=np.vstack(concat.rewards).reshape(-1))


def format_label(label: str) -> str:
    tokens = label.split('_')
    return ' '.join([t.capitalize() for t in tokens])


def make_model(model_path: str) -> StandardModel:
    save_folder, model_file = os.path.split(model_path)

    model_name = extract_model_name(model_file)
    assert model_name is not None, f'Could not extract name from file: {model_file}'

    # Extract hyperparameters
    hypers_name = HYPERS_PATH.format(model_name)
    hypers_path = os.path.join(save_folder, hypers_name)
    hypers = HyperParameters.create_from_file(hypers_path)

    # Build model and restore trainable parameters
    model = get_model(hypers, save_folder=save_folder, is_train=False)
    model.restore(name=model_name, is_train=False, is_frozen=False)

    return model
