import tensorflow as tf
import numpy as np
from collections import namedtuple
from typing import List
from argparse import ArgumentParser

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from threshold_optimization.optimize_thresholds import get_serialized_info
from controllers.logistic_regression_controller import fetch_model_states, POWER
from layers.basic import mlp
from utils.file_utils import extract_model_name
from utils.tfutils import get_activation
from utils.constants import BIG_NUMBER, SMALL_NUMBER


RolloutResult = namedtuple('RolloutResult', ['actions', 'rewards', 'observations', 'accuracy', 'avg_power'])


class ControllerPolicy:

    def __init__(self, batch_size: int, seq_length: int, input_units: int, hidden_units: int, activation: str, learning_rate: float, budget: float, budget_factor: float):
        self._batch_size = batch_size
        self._seq_length = seq_length
        self._input_units = input_units
        self._hidden_units = hidden_units
        self._activation = activation
        self._learning_rate = learning_rate
        self._budget = budget
        self._budget_factor = budget_factor
        self._is_made = False

        self._sess = tf.Session(graph=tf.Graph())
        
    def make(self):
        """
        Builds the computational graph for the policy.
        """
        with self._sess.graph.as_default():
            # Create the placeholders
            self._inputs = tf.placeholder(dtype=tf.float32,
                                          shape=[None, self._input_units],  # [B, D]
                                          name='input-placeholder')
            self._actions = tf.placeholder(dtype=tf.int32,
                                           shape=[None],  # [B]
                                           name='actions-placeholder')
            self._rewards = tf.placeholder(dtype=tf.float32,
                                           shape=[None],  # [B]
                                           name='reward-placeholder')

            # Create the computation graph. This is dense network with a single hidden layer.
            # The output is a [B, 2] tensor of log probabilities.
            logits, _ = mlp(inputs=self._inputs,
                            output_size=2,
                            hidden_sizes=[self._hidden_units],
                            activations=self._activation,
                            name='policy-network',
                            should_activate_final=False,
                            should_bias_final=True)
            self._logits = logits

            self._policy_sample = tf.multinomial(logits=logits, num_samples=1, seed=42, name='policy-sample')

            log_prob = tf.log(tf.nn.softmax(logits, axis=-1))

            # Get the log probability from the selected actions
            batch_idx = tf.range(start=0, limit=tf.shape(self._inputs)[0])  # [B]
            episode_idx = batch_idx * tf.shape(log_prob)[1]  # [B]
            action_idx = episode_idx + self._actions
            action_logits = tf.gather(tf.reshape(log_prob, [-1]), action_idx)  # [B]

            self._action_logits = action_logits

            # Calculate the surrogate loss function
            self._loss = -1 * tf.reduce_sum(action_logits * self._rewards)

            # Create the training step
            # self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
            self._optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate)
            self._train_step = self._optimizer.minimize(self._loss)

            self._is_made = True

    def init(self):
        with self._sess.graph.as_default():
            self._sess.run(tf.global_variables_initializer())

    def rollout(self, inputs: np.ndarray, labels: np.ndarray) -> RolloutResult:
        """
        Rolls out the policy for one episode on each batch element.

        Args:
            inputs: A [B, T - 1, D] array of input features for each time step (T) and batch element (B). We omit
                the last time step because there is no choice at the last step.
            labels: A [B, T] array of labels for each sequence element. We use these to derive the reward.
        Returns:
            A tuple containing the actions, rewards, and observations for this rollout.
        """
        assert self._is_made, 'Must make the model first'

        # Sample the policy. We do this in one tensorflow operation for efficiency.
        reshaped_inputs = inputs.reshape(-1, inputs.shape[-1])  # [B * (T - 1), D]

        with self._sess.graph.as_default():
            action_samples = self._sess.run(self._policy_sample, feed_dict={self._inputs: reshaped_inputs})

        episode_action_samples = action_samples.reshape(inputs.shape[0], inputs.shape[1])  # [B, T - 1]
       
        # Get the number of actions from each episode. We rely on the argmax tie-breaking scheme of selecting the
        # first element
        num_actions = np.argmax(episode_action_samples, axis=-1)  # [B]
        num_actions = np.where(np.logical_and(episode_action_samples[:, 0] == 0, num_actions == 0), self._seq_length - 1, num_actions)

        # Get the reward based on the correct / incorrect classification
        batch_idx = np.arange(labels.shape[0])
        action_labels = labels[batch_idx, num_actions]
        episode_rewards = 2 * action_labels - 1  # [B] array of -1 / 1 rewards for episode

        # Clip the number of actions to avoid the last index
        clipped_actions = np.clip(num_actions + 1, a_min=1, a_max=self._seq_length - 1)
        episode_rewards = np.repeat(episode_rewards, clipped_actions, axis=0)

        # Get the observations and actions for the sampled episodes
        observations: List[np.ndarray] = []
        actions: List[int] = []
        for batch_idx in range(episode_action_samples.shape[0]):
            for seq_idx in range(episode_action_samples.shape[1]):
                if seq_idx <= num_actions[batch_idx]:
                    observations.append(inputs[batch_idx, seq_idx])
                    actions.append(episode_action_samples[batch_idx, seq_idx])

        # Get the power for each episode
        episode_power = POWER[num_actions]
        episode_power = np.repeat(episode_power, clipped_actions, axis=0)
        episode_rewards = episode_rewards - self._budget_factor * np.clip(episode_power - self._budget, a_min=0.0, a_max=None)

        # Compute the average power
        seq_count = np.bincount(num_actions, minlength=self._seq_length)
        normalized_seq_count = seq_count / (np.sum(seq_count) + SMALL_NUMBER)
        avg_power = np.sum(normalized_seq_count * POWER)

        return RolloutResult(rewards=episode_rewards, actions=np.array(actions), observations=np.array(observations), accuracy=np.average(action_labels), avg_power=avg_power)
    
    def train(self, inputs: np.ndarray, labels: np.ndarray, max_epochs: int, patience: int):
        """
        Trains the policy controller.

        Args:
            inputs: A [N, T, D] array of input features (D) for each timestep (T) and sample (N)
            labels: A [B, T] array of labels
            max_epochs: The maximum number of epochs to train for.
        """
        truncated_inputs = inputs[:, 0:-1, :]  # [N, T-1, D]
        sample_idx = np.arange(inputs.shape[0])  # [N]
        rand = np.random.RandomState(seed=32)

        best_accuracy = 0.0
        early_stopping_counter = 0

        for epoch in range(max_epochs):
            print('===== Starting Epoch: {0} ====='.format(epoch))

            rand.shuffle(sample_idx)

            epoch_inputs = truncated_inputs[sample_idx]
            epoch_labels = labels[sample_idx]

            batch_loss: List[float] = []
            batch_accuracy: List[float] = []
            batch_power: List[float] = []
            for batch_num, batch_idx in enumerate(range(0, sample_idx.shape[0] - self._batch_size + 1, self._batch_size)):
                start, end = batch_idx, batch_idx + self._batch_size

                # Perform a policy rollout
                rollout_result = self.rollout(inputs=epoch_inputs[start:end],
                                              labels=epoch_labels[start:end])

                batch_obs = rollout_result.observations
                batch_rewards = rollout_result.rewards
                batch_actions = rollout_result.actions
                avg_power = rollout_result.avg_power

                # Subtract the baseline from the reward to lower variance
                batch_rewards = (batch_rewards - np.average(batch_rewards)) / (np.std(batch_rewards) + SMALL_NUMBER)

                # Place penalty on power usage
                # batch_rewards = batch_rewards - self._budget_factor * np.clip(avg_power - self._budget, a_min=0, a_max=None)

                # Run the training step and calculate the loss
                feed_dict = {self._inputs: batch_obs, self._rewards: batch_rewards, self._actions: batch_actions}
                result = self._sess.run({'logits': self._logits, 'loss': self._loss, 'train_step': self._train_step}, feed_dict=feed_dict)

                batch_loss.append(result['loss'])
                batch_accuracy.append(rollout_result.accuracy)
                batch_power.append(avg_power)

                acc = np.average(batch_accuracy)
                avg_loss = np.average(batch_loss)

                print('Completed Batch {0}. Avg loss so far: {1:.5f}. Accuracy so far: {2:.5f}. Avg power so far: {3:.5f}'.format(batch_num, avg_loss, acc, avg_power), end='\r')

            batch_acc = np.average(batch_accuracy)
            
            if batch_acc > best_accuracy:
                early_stopping_counter = 0
                best_accuracy = batch_acc
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print('\nConverged.')
                break

            print()

    def test(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Evaluates the current policy on the given testing set.
        """
        truncated_inputs = inputs[:, 0:-1, :]

        num_correct: List[int] = []
        seq_counts = np.zeros(shape=(self._seq_length,))

        num_samples = inputs.shape[0]
        for batch_num, batch_idx in enumerate(range(0, num_samples, self._batch_size)):
            start, end = batch_idx, batch_idx + self._batch_size

            batch_inputs = truncated_inputs[start:end]
            batch_labels = labels[start:end]

            # Run the training step and calculate the loss
            feed_dict = {self._inputs: batch_inputs.reshape(-1, inputs.shape[-1])}
            result = self._sess.run({'logits': self._logits}, feed_dict=feed_dict)

            # Reshape the logits for each unique episode
            batch_logits = result['logits'].reshape(-1, self._seq_length - 1, 2)

            # Get the actions for each sequence element
            batch_actions = np.argmax(batch_logits, axis=-1)  # [B, T - 1]
            selected_elements = np.argmax(batch_actions, axis=1)  # [B]
            selected_elements = np.where(np.all(batch_actions == 0, axis=1), self._seq_length - 1, selected_elements)

            batch_indices = np.arange(batch_labels.shape[0])
            selected_labels = batch_labels[batch_indices, selected_elements]  # [B]
            num_correct.extend(selected_labels)    

            # Compute the average power
            selection_counts = np.bincount(selected_elements, minlength=self._seq_length)
            seq_counts += selection_counts 
            normalized_counts = seq_counts / (np.sum(seq_counts) + SMALL_NUMBER)
            avg_power = np.sum(POWER * normalized_counts)

            print('Accuracy so far: {0:.5f}. Avg power so far: {1:.5f}.'.format(np.average(num_correct), avg_power), end='\r')
            break
        print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    args = parser.parse_args()

    model, dataset, _ = get_serialized_info(args.model_path, dataset_folder=args.dataset_folder) 

    X_train, y_train, _, _ = fetch_model_states(model=model, dataset=dataset, series=DataSeries.TRAIN)

    policy = ControllerPolicy(batch_size=128,
                              seq_length=X_train.shape[1],
                              input_units=X_train.shape[2],
                              hidden_units=16,
                              activation='leaky_relu',
                              learning_rate=0.001,
                              budget=47,
                              budget_factor=0.01)
    policy.make()
    policy.init()

    # Train the policy
    policy.train(inputs=X_train, labels=y_train, max_epochs=100, patience=50)

    # Load the test data
    X_test, y_test, _, _ = fetch_model_states(model=model, dataset=dataset, series=DataSeries.TEST)
    policy.test(inputs=X_test, labels=y_test)
