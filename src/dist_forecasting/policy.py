import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from argparse import ArgumentParser
from collections import namedtuple, Counter
from typing import Dict, Any, Tuple, Union, List

from dataset.dataset import Dataset, DataSeries
from dataset.dataset_factory import get_dataset
from utils.constants import BIG_NUMBER, SMALL_NUMBER, PREDICTION, OPTIMIZER_OP, LOSS, INPUTS
from utils.file_utils import read_pickle_gz
from utils.np_utils import subsample_inputs
from dist_utils import make_model, RolloutResult, stack_rollouts
from rnn import ugrnn_cell, output_layer


ABSOLUTE_ERROR = 'absolute_error'


class Policy:

    def __init__(self, model_vars: Dict[str, np.ndarray]):
        self._state_size = model_vars['model/embedding-layer-kernel:0'].shape[-1]
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        self._model_vars = model_vars
        self._ops: Dict[str, tf.Tensor] = dict()  # Holds all operations

    def make(self):
        with self._sess.graph.as_default():
           
            # Input placeholders
            self._input_ph = tf.compat.v1.placeholder(shape=(None, self._state_size + 1),  # [B, D]
                                                      dtype=tf.float32,
                                                      name='input-ph')
            self._output_ph = tf.compat.v1.placeholder(shape=(None),  # [B]
                                                       dtype=tf.float32,
                                                       name='output-ph')

            # Compute prediction, [B, 1]
            hidden = tf.compat.v1.layers.dense(inputs=self._input_ph,
                                               units=20,
                                               activation=tf.nn.relu,
                                               name='hidden')
            pred = tf.compat.v1.layers.dense(inputs=hidden,
                                             units=1,
                                             activation=tf.math.sigmoid,
                                             name='readout')
            pred = tf.squeeze(pred, axis=-1)  # [B]
            self._ops[PREDICTION] = pred  # [B]

            # Calculate the loss, Scalar
            self._ops[LOSS] = tf.reduce_sum(tf.square(pred - self._output_ph))
            
            # Calculate the Absolute Error
            self._ops[ABSOLUTE_ERROR] = tf.reduce_sum(tf.abs(pred - self._output_ph))

            # Create the update step
            self._optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
            self._ops[OPTIMIZER_OP] = self._optimizer.minimize(self._ops[LOSS])

    def init(self):
        with self._sess.graph.as_default():
            self._sess.run(tf.compat.v1.global_variables_initializer())

    def execute(self, ops_to_run: Union[List[str], str], feed_dict: Dict[tf.compat.v1.placeholder, np.ndarray]) -> Dict[str, np.ndarray]:

        # Convert operations to a list of names
        ops_list = [ops_to_run] if isinstance(ops_to_run, str) else ops_to_run

        with self._sess.graph.as_default():
            ops_dict = {op_name: self._ops[op_name] for op_name in ops_list}
            result = self._sess.run(ops_dict, feed_dict=feed_dict)
        
        return result

    def rollout(self, inputs: np.ndarray) -> RolloutResult:
        """
        Performs a policy rollout on the given sequence
        of input features.

        inputs: A [T, K] array of input features to the RNN
        """
        seq_length = inputs.shape[0]

        # Generate the random sample
        subseq_length = np.random.randint(low=1, high=seq_length)  # T'
        subseq_indices = np.random.choice(a=seq_length, size=subseq_length, replace=False)

        # Always add in the first sample
        if (0 not in subseq_indices): 
            subseq_indices[-1] = 0

        subseq_indices = np.sort(subseq_indices)

        state = np.zeros(shape=(1, self._state_size))  # [1, D]
        states: List[np.ndarray] = []

        inputs = np.expand_dims(inputs, axis=0)  # [1, T, K]

        for i, index in enumerate(subseq_indices):
            if i == len(subseq_indices) - 1:
                gap = seq_length - index
            else:
                gap = subseq_indices[i + 1] - index

            # Perform the RNN transition
            step_inputs = inputs[:, index, :]  # [1, K]
            state = ugrnn_cell(input_features=step_inputs,
                               state=state,
                               model_vars=self._model_vars)

            policy_state = np.concatenate([state, [[gap]]], axis=-1)  # [1, D + 1]
            states.append(policy_state)

        # Compute the prediction
        pred = output_layer(states=state, model_vars=self._model_vars)
        reward = np.max(pred)  # TODO: Support for configurable metric

        rewards = [reward for _ in states]

        return RolloutResult(states=states, rewards=rewards)

    def train(self, dataset: Dataset, series: DataSeries, metadata: Dict[str, Any], num_epochs: int, batch_size: int):

        TRAIN_OPS = [LOSS, ABSOLUTE_ERROR, OPTIMIZER_OP, PREDICTION]

        for epoch in range(num_epochs):

            print('==========')
            print('Epoch {0}/{1}'.format(epoch + 1, num_epochs))
            print('==========')

            train_generator = dataset.minibatch_generator(series=series,
                                                          metadata=metadata,
                                                          batch_size=batch_size,
                                                          should_shuffle=True)

            train_loss, train_error, train_samples = 0, 0, 0
            for train_idx, batch in enumerate(train_generator):
                # Perform a random rollout on the sequence elements to generate
                # a training batch
                batch_rollouts = [self.rollout(inputs) for inputs in batch[INPUTS]]
                train_batch = stack_rollouts(batch_rollouts)

                feed_dict = {
                    self._input_ph: train_batch.states,
                    self._output_ph: train_batch.rewards
                }

                train_result = self.execute(TRAIN_OPS, feed_dict=feed_dict)
                loss = train_result[LOSS]
                abs_error = train_result[ABSOLUTE_ERROR]

                train_loss += loss
                train_error += abs_error
                train_samples += len(train_batch.rewards)

                #if epoch == (num_epochs - 1) and train_idx == 0:
                #    print('Predictions: {0}'.format(train_result[PREDICTION]))
                #    print('Expected: {0}'.format(train_batch.rewards))
                #    print('==========')

            print('Avg Train Loss: {0:.5f}, Avg Train Error: {1:.5f}'.format(train_loss / train_samples, train_error / train_samples))
 
            val_generator = dataset.minibatch_generator(series=series,
                                                        metadata=metadata,
                                                        batch_size=batch_size,
                                                        should_shuffle=False)

            val_loss, val_error, val_samples = 0, 0, 0
            for val_idx, val_batch in enumerate(val_generator):
                # Perform a random rollout on the sequence elements to generate a batch
                batch_rollouts = [self.rollout(inputs) for inputs in batch[INPUTS]]
                val_batch = stack_rollouts(batch_rollouts)

                feed_dict = {
                    self._input_ph: val_batch.states,
                    self._output_ph: val_batch.rewards
                }

                val_result = self.execute([LOSS, PREDICTION, ABSOLUTE_ERROR], feed_dict=feed_dict)
                loss = val_result[LOSS]
                abs_error = val_result[ABSOLUTE_ERROR]
                pred = val_result[PREDICTION]

                #if epoch == (num_epochs - 1) and val_idx == 0:
                #    print('Predictions: {0}'.format(pred))
                #    print('Expected: {0}'.format(val_batch.rewards))
                #    print('==========')

                val_loss += loss
                val_error += abs_error
                val_samples += len(val_batch.rewards)

            print('Avg Val Loss: {0:.5f}, Avg Val Error: {1:.5f}'.format(val_loss / val_samples, val_error / val_samples))

    def adaptive_inference(self, dataset: Dataset, series: DataSeries, metadata: Dict[str, Any], target: float) -> Tuple[float, float, Counter]:
        generator = dataset.minibatch_generator(series=series,
                                                metadata=metadata,
                                                batch_size=1,
                                                should_shuffle=False)

        pos_distribution: Counter = Counter()
    
        num_correct, num_samples, num_updates = 0.0, 0, 0.0
        for idx, batch in enumerate(generator):
            inputs = np.expand_dims(batch['inputs'][0], axis=0)  # [1, T, D]
            state = np.zeros(shape=(1, self._state_size))  # [1, D]

            pos = 0
            while pos < inputs.shape[1]:
                pos_distribution[pos] += 1

                # Perform the RNN transition
                step_inputs = inputs[:, pos, :]  # [1, K]
                state = ugrnn_cell(input_features=step_inputs,
                                   state=state,
                                   model_vars=self._model_vars)

                num_updates += 1

                # Roll out over multiple gaps
                policy_action = 1

                if pos < inputs.shape[1] - 1:
                    policy_states: List[np.ndarray] = []
                    for gap in range(1, inputs.shape[1] - pos + 1):
                        policy_state = np.concatenate([state, [[gap]]], axis=-1)  # [1, D + 1]
                        policy_states.append(policy_state)

                    feed_dict = { self._input_ph: np.vstack(policy_states) }
                    policy_result = self.execute([PREDICTION], feed_dict=feed_dict)

                    diff = np.abs(policy_result[PREDICTION] - target)
                    policy_action = np.argmin(diff) + 1

                pos += policy_action

            # Compute the prediction
            pred_dist = output_layer(states=state, model_vars=self._model_vars)
            pred = np.argmax(pred_dist[0])

            # Compare to the output
            is_correct = float(int(pred) == int(batch['output'][0]))

            num_correct += is_correct
            num_samples += 1

        avg_updates = num_updates / num_samples
        accuracy = num_correct / num_samples
        print('Accuracy: {0:.4f}'.format(accuracy))
        print('Avg Updates: {0:.4f}'.format(avg_updates))

        for pos in pos_distribution.keys():
            pos_distribution[pos] /= num_samples
            pos_distribution[pos] *= 100.0

        print(pos_distribution)

        return accuracy, avg_updates, pos_distribution

    def random_inference(self, dataset: Dataset, series: DataSeries, metadata: Dict[str, Any], target: float) -> float:
        generator = dataset.minibatch_generator(series=series,
                                                metadata=metadata,
                                                batch_size=1,
                                                should_shuffle=False)

        updates_per_seq = int(math.ceil(target))

        rand = np.random.RandomState(seed=132)

        num_correct, num_samples = 0.0, 0
        for batch in generator:
            inputs = batch[INPUTS][0]  # [T, K]

            # Sample the input sequence down to the given length
            indices = rand.choice(inputs.shape[0], size=updates_per_seq, replace=False)  # [T']
            indices = np.sort(indices)  # [T']

            # Collect the sub-sampled batch
            sampled_inputs = inputs[indices, :]  # [T', K]

            state = np.zeros(shape=(1, self._state_size))  # [1, D]
            for sample in sampled_inputs:
                state = ugrnn_cell(input_features=np.expand_dims(sample, axis=0),
                                   state=state,
                                   model_vars=self._model_vars)

            pred_dist = output_layer(states=state, model_vars=self._model_vars)
            pred = int(np.argmax(pred_dist[0]))  # Scalar

            expected = int(batch['output'][0])
            num_correct += float(expected == pred)
            num_samples += 1

        accuracy = num_correct / num_samples
        print('Accuracy: {0:.4f}, Updates per Seq: {1}'.format(accuracy, updates_per_seq))

        return accuracy


def plot_distribution(pos_distribution: Counter,
                      target: float,
                      accuracy: float,
                      avg_updates: float,
                      random_accuracy: float):

    WIDTH = 0.5

    with plt.style.context('seaborn-ticks'):

        fix, ax = plt.subplots()

        # Plot values in a bar chart
        xs = list(sorted(pos_distribution.keys()))
        values = [pos_distribution[pos] for pos in xs]
        ax.bar(x=xs, height=values, width=WIDTH)

        ax.set_title('Percentage using Each Sequence Element with Target {0:.2f}'.format(target))
        ax.set_xlabel('Element Index')
        ax.set_ylabel('Percentage of Samples')

        # Annotate plot with accuracy and avg # Updates
        random_updates = int(math.ceil(avg_updates))
        mid = np.average(xs)
        ax.text(x=mid, y=100, s='Adaptive Accuracy: {0:.4f}'.format(accuracy))
        ax.text(x=mid, y=95, s='Adaptive Updates: {0:.3f}'.format(avg_updates))
        ax.text(x=mid, y=90, s='Random Accuracy: {0:.4f}'.format(random_accuracy))
        ax.text(x=mid, y=85, s='Random Updates: {0}'.format(random_updates))

        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-folder', type=str, required=True)
    args = parser.parse_args()

    model = make_model(args.model_path)
    dataset = get_dataset(dataset_type='randomized', data_folder=args.data_folder)

    model_vars = read_pickle_gz(args.model_path)
    metadata = model.metadata

    policy = Policy(model_vars=model_vars)

    policy.make()
    policy.init()

    policy.train(dataset=dataset,
                 series=DataSeries.VALID,
                 metadata=metadata,
                 num_epochs=20,
                 batch_size=16)

    print('==========')
    print('Adaptive Inference:')

    target = 0.6
    accuracy, avg_updates, dist = policy.adaptive_inference(dataset=dataset,
                                                            series=DataSeries.TEST,
                                                            metadata=metadata,
                                                            target=target)

    print('Random Inference:')
    random_acc = policy.random_inference(dataset=dataset,
                                         series=DataSeries.TEST,
                                         metadata=metadata,
                                         target=avg_updates)

    plot_distribution(pos_distribution=dist,
                      target=target,
                      accuracy=accuracy,
                      avg_updates=avg_updates,
                      random_accuracy=random_acc)
