import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, Any, Tuple

from dataset.dataset import Dataset, DataSeries
from dataset.dataset_factory import get_dataset
from utils.constants import BIG_NUMBER, SMALL_NUMBER
from utils.file_utils import read_pickle_gz
from dist_utils import make_model
from rnn import ugrnn_cell, output_layer


RolloutResult = namedtuple('RolloutResult', ['actions', 'positions', 'states', 'rewards'])


class Policy:

    def __init__(self, model_vars: Dict[str, np.ndarray], seq_length: int):
        self._state_size = model_vars['model/embedding-layer-kernel:0'].shape[-1]
        self._seq_length = seq_length
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        self._model_vars = model_vars

    def make(self):
        with self._sess.graph.as_default():
           
            # Input placeholders
            self._input_ph = tf.compat.v1.placeholder(shape=(None, self._state_size),
                                                      dtype=tf.float32,
                                                      name='input-ph')
            self._pos_ph = tf.compat.v1.placeholder(shape=(None, 1),
                                                    dtype=tf.int32,
                                                    name='pos-ph')

            # Placeholders for actions and rewards
            self._actions_ph = tf.compat.v1.placeholder(shape=(None),
                                                        dtype=tf.int32,
                                                        name='actions-ph')
            self._rewards_ph = tf.compat.v1.placeholder(shape=(None),
                                                        dtype=tf.float32,
                                                        name='rewards-ph')

            # Compute the log probabilities, [B, T]
            logits = tf.compat.v1.layers.dense(inputs=self._input_ph,
                                               units=self._seq_length - 1,
                                               activation=None,
                                               name='policy')

            # Calculate log probabilities via a mask over non-existent positions
            seq_idx = tf.expand_dims(tf.range(start=0, limit=self._seq_length - 1), axis=0)  # [1, T]
            mask = tf.cast((self._seq_length - self._pos_ph) <= seq_idx, dtype=tf.float32) * -BIG_NUMBER  # [B, T]
            masked_logits = logits + mask  # [B, T]

            # Create operation to sample the policy
            self._logits = mask
            self._sample = tf.random.categorical(logits=masked_logits, num_samples=1, seed=9634)

            # Normalize the action log probabilities
            probs = tf.nn.softmax(masked_logits, axis=-1)  # [B, T]
            
            # Compute the (normalized) log probs (for the surrogate loss function)
            action_logits = tf.math.log(probs + SMALL_NUMBER)  # [B, T]
            action_logits = tf.reshape(action_logits, (-1, ))  # [B * T]

            # Get indices corresponding to each action
            batch_size, seq_length = tf.shape(logits)[0], tf.shape(logits)[1]
            indices = tf.range(start=0, limit=batch_size) * seq_length + self._actions_ph  # [B]

            # Get the log probabilities for the actions
            self._action_logits_before = action_logits
            action_logits = tf.gather(action_logits, indices)
            self._action_logits_after = action_logits
 
            # Calculate the surrogate loss
            self._loss = -tf.reduce_sum(action_logits * self._rewards_ph)

            # Create the update step
            self._optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
            self._train_op = self._optimizer.minimize(self._loss)

    def init(self):
        with self._sess.graph.as_default():
            self._sess.run(tf.compat.v1.global_variables_initializer())

    def sample(self, states: np.ndarray, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(states.shape) != 2:
            states = np.expand_dims(states, axis=0)  # [1, D]

        with self._sess.graph.as_default():
            feed_dict = {
                self._input_ph: states,
                self._pos_ph: positions
            }

            actions, logits = self._sess.run([self._sample, self._logits], feed_dict=feed_dict)  # [B]

        return actions, logits

    def rollout(self, inputs: np.ndarray) -> RolloutResult:
        """
        Performs a policy rollout on the given sequence
        of input features.

        inputs: A [T, K] array of input features to the RNN
        """
        inputs = np.expand_dims(inputs, axis=0)  # [1, T, K]
        state = np.zeros(shape=(1, self._state_size))  # [1, D]

        states: List[np.ndarray] = []
        actions: List[int] = []
        positions: List[int] = []

        pos = 0
        while pos < self._seq_length:
            # Perform the RNN transition
            step_inputs = inputs[:, pos, :]  # [1, K]
            state = ugrnn_cell(input_features=step_inputs,
                               state=state,
                               model_vars=self._model_vars)

            states.append(state)
            positions.append(pos)

            # Sample the policy, [1, 1]
            action, _ = self.sample(states=state, positions=[[pos]])
            actions.append(action[0, 0])

            # Update the position
            pos += (action[0, 0] + 1)

        # Compute the prediction
        pred = output_layer(states=state, model_vars=self._model_vars)
        reward = np.max(pred)

        rewards = [reward for _ in positions]

        return RolloutResult(states=states, actions=actions, rewards=rewards, positions=positions)

    def train_step(self, rollout_result: RolloutResult) -> float:
        with self._sess.graph.as_default():
            feed_dict = {
                self._input_ph: np.vstack(rollout_result.states),
                self._pos_ph: np.expand_dims(rollout_result.positions, axis=-1),
                self._actions_ph: rollout_result.actions,
                self._rewards_ph: rollout_result.rewards
            }

            ops = {
                'loss': self._loss,
                'opt': self._train_op,
                'logits_before': self._action_logits_before,
                'logits_after': self._action_logits_after

            }
            result = self._sess.run(ops, feed_dict=feed_dict)

        return result['loss']

    def train(self, dataset: Dataset, series: DataSeries, metadata: Dict[str, Any], num_epochs: int, batch_size: int):

        for epoch in range(num_epochs):

            generator = dataset.minibatch_generator(series=series,
                                                    metadata=metadata,
                                                    batch_size=batch_size,
                                                    should_shuffle=True)

            total_loss = 0
            total_reward = 0
            for idx, batch in enumerate(generator):
                batch_rollout = RolloutResult(states=[], actions=[], rewards=[], positions=[])

                for inputs in batch['inputs']:
                    rollout = self.rollout(inputs=batch['inputs'][0])

                    total_reward += rollout.rewards[0]

                    batch_rollout.states.extend(rollout.states)
                    batch_rollout.actions.extend(rollout.actions)
                    batch_rollout.rewards.extend(rollout.rewards)
                    batch_rollout.positions.extend(rollout.positions)

                loss = self.train_step(rollout_result=batch_rollout)
                total_loss += loss

                # print('Loss: {0:.5f}'.format(loss), end='\r')
                # print(loss)
                # print(result)

            total_samples = idx * batch_size
            print('Loss: {0}'.format(total_loss / total_samples))
            print('Reward: {0}'.format(total_reward / total_samples))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-folder', type=str, required=True)
    args = parser.parse_args()

    model = make_model(args.model_path)
    dataset = get_dataset(dataset_type='randomized', data_folder=args.data_folder)

    model_vars = read_pickle_gz(args.model_path)
    metadata = model.metadata

    policy = Policy(model_vars=model_vars, seq_length=metadata['seq_length'])

    policy.make()
    policy.init()

    policy.train(dataset=dataset,
                 series=DataSeries.VALID,
                 metadata=metadata,
                 num_epochs=10,
                 batch_size=16)
