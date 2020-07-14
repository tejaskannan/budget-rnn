import tensorflow as tf
import numpy as np
import os.path
from collections import namedtuple
from typing import List
from argparse import ArgumentParser

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from threshold_optimization.optimize_thresholds import get_serialized_info
from controllers.logistic_regression_controller import fetch_model_states, POWER
from layers.basic import mlp
from utils.file_utils import extract_model_name, read_by_file_suffix, save_by_file_suffix
from utils.tfutils import get_activation
from utils.constants import BIG_NUMBER, SMALL_NUMBER


RolloutResult = namedtuple('RolloutResult', ['actions', 'rewards', 'observations', 'accuracy', 'avg_power'])


class ControllerPolicy:

    def __init__(self,
                 batch_size: int,
                 seq_length: int,
                 input_units: int,
                 hidden_units: int,
                 activation: str,
                 learning_rate: float,
                 budget: float,
                 save_folder: str,
                 name: str):
        self._batch_size = batch_size
        self._seq_length = seq_length
        self._input_units = input_units
        self._hidden_units = hidden_units
        self._activation = activation
        self._learning_rate = learning_rate
        self._budget = budget
        self._save_folder = save_folder
        self._name = name
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
            self._penalty = tf.placeholder(dtype=tf.float32,
                                           shape=[None],  # [B]
                                           name='penalty-placeholder')

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
            regularized_rewards = (self._rewards - tf.reduce_mean(self._rewards)) - self._penalty
            self._loss = -1 * tf.reduce_sum(action_logits * regularized_rewards)

            # Create the training step
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
        episode_rewards = np.where(episode_rewards > 0, episode_rewards, 25 * episode_rewards)

        # Get the observations and actions for the sampled episodes
        observations: List[np.ndarray] = []
        actions: List[int] = []
        for batch_idx in range(episode_action_samples.shape[0]):
            for seq_idx in range(episode_action_samples.shape[1]):
                if seq_idx <= num_actions[batch_idx]:
                    observations.append(inputs[batch_idx, seq_idx])
                    actions.append(episode_action_samples[batch_idx, seq_idx])

        # Get the power for each episode and penalize violations
        # episode_power = POWER[num_actions]
        # episode_power = np.repeat(episode_power, clipped_actions, axis=0)
        # episode_rewards = episode_rewards - 2 * np.clip((episode_power - self._budget) / self._budget, a_min=0.0, a_max=None)

        # Compute the average power
        seq_count = np.bincount(num_actions, minlength=self._seq_length)
        normalized_seq_count = seq_count / (np.sum(seq_count) + SMALL_NUMBER)
        avg_power = np.sum(normalized_seq_count * POWER)

        return RolloutResult(rewards=episode_rewards, actions=np.array(actions), observations=np.array(observations), accuracy=np.average(action_labels), avg_power=avg_power)
    
    def train(self, train_inputs: np.ndarray, train_labels: np.ndarray, valid_inputs: np.ndarray, valid_labels: np.ndarray, max_epochs: int, patience: int):
        """
        Trains the policy controller.

        Args:
            train_inputs: A [N, T, D] array of input features (D) for each timestep (T) and sample (N)
            train_labels: A [N, T] array of labels
            valid_inputs: A [M, T, D] array of input features for the validation set
            valid_labels: A [M ,T] array of validation labels
            max_epochs: The maximum number of epochs to train for
            patience: Patience for early stopping
        """
        truncated_train_inputs = train_inputs[:, 0:-1, :]  # [N, T-1, D]
        truncated_valid_inputs = valid_inputs[:, 0:-1, :]  # [M, T-1, D]

        num_train = truncated_train_inputs.shape[0]
        num_valid = truncated_valid_inputs.shape[0]

        sample_idx = np.arange(num_train)  # [N]
        rand = np.random.RandomState(seed=32)  # Set random state for reproducible results

        best_accuracy = 0.0
        early_stopping_counter = 0

        for epoch in range(max_epochs):
            print('===== Starting Epoch: {0} ====='.format(epoch))

            # Shuffle the training set
            rand.shuffle(sample_idx)
            epoch_train_inputs = truncated_train_inputs[sample_idx, :, :]
            epoch_train_labels = train_labels[sample_idx, :]

            train_batch_loss: List[float] = []
            train_batch_accuracy: List[float] = []
            train_batch_power: List[float] = []
            for batch_num, batch_idx in enumerate(range(0, num_train - self._batch_size + 1, self._batch_size)):
                start, end = batch_idx, batch_idx + self._batch_size

                # Perform a policy rollout
                rollout_result = self.rollout(inputs=epoch_train_inputs[start:end, :, :],
                                              labels=epoch_train_labels[start:end, :])

                batch_obs = rollout_result.observations
                batch_rewards = rollout_result.rewards
                batch_actions = rollout_result.actions
                avg_power = rollout_result.avg_power

                # Normalize the batch rewards
                batch_rewards = (batch_rewards - np.average(batch_rewards)) / (np.std(batch_rewards) + SMALL_NUMBER)

                # Apply penalty for power
                penalty = 10 * np.clip(avg_power - self._budget, a_min=0.0, a_max=None)
                batch_penalty = np.full(shape=batch_rewards.shape, fill_value=penalty)

                # Run the training step and calculate the loss
                with self._sess.graph.as_default():
                    feed_dict = {self._inputs: batch_obs, self._rewards: batch_rewards, self._penalty: batch_penalty, self._actions: batch_actions}
                    result = self._sess.run({'logits': self._logits, 'loss': self._loss, 'train_step': self._train_step}, feed_dict=feed_dict)

                train_batch_loss.append(result['loss'])
                train_batch_accuracy.append(rollout_result.accuracy)
                train_batch_power.append(avg_power)

                acc = np.average(train_batch_accuracy)
                avg_loss = np.average(train_batch_loss)

                print('Completed Training Batch {0}. Avg loss so far: {1:.5f}. Accuracy so far: {2:.5f}. Avg power so far: {3:.5f}'.format(batch_num, avg_loss, acc, avg_power), end='\r')
            print()

            valid_batch_accuracy: List[float] = []
            valid_seq_counts = np.zeros(shape=(self._seq_length, ))
            for batch_num, batch_idx in enumerate(range(0, num_valid, self._batch_size)):
                start, end = batch_idx, batch_idx + self._batch_size

                batch_inputs = valid_inputs[start:end, :, :]
                batch_labels = valid_labels[start:end, :]

                # Sample the policy
                with self._sess.graph.as_default():
                    feed_dict = {self._inputs: batch_inputs.reshape(-1, valid_inputs.shape[-1])}
                    action_samples = self._sess.run(self._policy_sample, feed_dict=feed_dict)

                episode_action_samples = action_samples.reshape(batch_inputs.shape[0], batch_inputs.shape[1])  # [B, T - 1]
       
                # Get the number of actions from each episode. We rely on the argmax tie-breaking scheme of selecting the
                # first element
                num_actions = np.argmax(episode_action_samples, axis=-1)  # [B]
                selected_elements = np.where(np.logical_and(episode_action_samples[:, 0] == 0, num_actions == 0), self._seq_length - 1, num_actions)

                batch_indices = np.arange(batch_labels.shape[0])
                selected_labels = batch_labels[batch_indices, selected_elements]
                valid_batch_accuracy.extend(selected_labels)
                accuracy = np.average(valid_batch_accuracy)

                # Compute the average power
                selection_counts = np.bincount(selected_elements, minlength=self._seq_length)
                valid_seq_counts += selection_counts 
                normalized_counts = valid_seq_counts / (np.sum(valid_seq_counts) + SMALL_NUMBER)
                avg_power = np.sum(POWER * normalized_counts)

                print('Completed Validation Batch {0}. Accuracy so far: {1:.5f}. Avg power so far: {2:.5f}.'.format(batch_num, accuracy, avg_power), end='\r')
            print()

            valid_accuracy = np.average(valid_batch_accuracy)
            if valid_accuracy >= best_accuracy:
                best_accuracy = valid_accuracy
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            print('Saving...')
            self.save()

            if early_stopping_counter >= patience:
                print('Converged.')
                break

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
            #result = self._sess.run({'logits': self._logits}, feed_dict=feed_dict)

            ## Reshape the logits for each unique episode
            #batch_logits = result['logits'].reshape(-1, self._seq_length - 1, 2)

            ## Get the actions for each sequence element using a greedy strategy
            #batch_actions = np.argmax(batch_logits, axis=-1)  # [B, T - 1]
            #selected_elements = np.argmax(batch_actions, axis=1)  # [B]
            #selected_elements = np.where(np.all(batch_actions == 0, axis=1), self._seq_length - 1, selected_elements)

            #batch_indices = np.arange(batch_labels.shape[0])
            #selected_labels = batch_labels[batch_indices, selected_elements]  # [B]
            #num_correct.extend(selected_labels)

            # Sample the policy
            with self._sess.graph.as_default():
                action_samples = self._sess.run(self._policy_sample, feed_dict=feed_dict)

            episode_action_samples = action_samples.reshape(batch_inputs.shape[0], batch_inputs.shape[1])  # [B, T - 1]
       
            # Get the number of actions from each episode. We rely on the argmax tie-breaking scheme of selecting the
            # first element
            num_actions = np.argmax(episode_action_samples, axis=-1)  # [B]
            selected_elements = np.where(np.logical_and(episode_action_samples[:, 0] == 0, num_actions == 0), self._seq_length - 1, num_actions)

            batch_indices = np.arange(batch_labels.shape[0])
            selected_labels = batch_labels[batch_indices, selected_elements]
            num_correct.extend(selected_labels)

            # Compute the average power
            selection_counts = np.bincount(selected_elements, minlength=self._seq_length)
            seq_counts += selection_counts 
            normalized_counts = seq_counts / (np.sum(seq_counts) + SMALL_NUMBER)
            avg_power = np.sum(POWER * normalized_counts)

            print('Accuracy so far: {0:.5f}. Avg power so far: {1:.5f}.'.format(np.average(num_correct), avg_power), end='\r')
        print()

        print(seq_counts)

    def save(self):
        # Get the model hyperparameters
        hypers = {
            'batch_size': self._batch_size,
            'seq_length': self._seq_length,
            'input_units': self._input_units,
            'hidden_units': self._hidden_units,
            'activation': self._activation,
            'learning_rate': self._learning_rate,
            'budget': self._budget
        }

        # Create the output path
        output_path = os.path.join(self._save_folder, 'model-policy-{0}-{1}.pkl.gz'.format(self._budget, self._name))

        # Get the trainable variables
        with self._sess.graph.as_default():
            trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            var_dict = self._sess.run({var.name: var for var in trainable_vars})

            save_dict = dict(hypers=hypers, variables=var_dict)
            save_by_file_suffix(save_dict, output_path)

    @classmethod
    def restore(cls, path: str):
        # Read the serialized information
        saved_dict = read_by_file_suffix(path)

        # Create the model
        hypers = saved_dict['hypers']
        policy = ControllerPolicy(batch_size=hypers['batch_size'],
                                  seq_length=hypers['seq_length'],
                                  input_units=hypers['input_units'],
                                  hidden_units=hypers['hidden_units'],
                                  activation=hypers['activation'],
                                  learning_rate=hypers['learning_rate'],
                                  budget=hypers['budget'])
        policy.make()

        # Read the saved variables
        var_dict = saved_dict['variables']

        # Restore variable values
        assign_ops = []
        trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARS)
        for var in trainable_vars:
            saved_value = vars_dict[var.name]
            assign_op = var.assign(saved_value, use_locking=True, read_value=False)
            assign_ops.append(assign_op)

        # Execute the assignment
        self._sess.run(assign_ops)
        return policy

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    args = parser.parse_args()

    save_folder, model_file_name = os.path.split(args.model_path)
    model_name = extract_model_name(model_file_name)

    model, dataset, _ = get_serialized_info(args.model_path, dataset_folder=args.dataset_folder) 

    X_train, y_train, _, _, _ = fetch_model_states(model=model, dataset=dataset, series=DataSeries.TRAIN)
    X_valid, y_valid, _, _, _ = fetch_model_states(model=model, dataset=dataset, series=DataSeries.VALID)

    y_max = np.max(y_train, axis=-1)  # [B]

    policy = ControllerPolicy(batch_size=64,
                              seq_length=X_train.shape[1],
                              input_units=X_train.shape[2],
                              hidden_units=16,
                              activation='leaky_relu',
                              learning_rate=0.001,
                              budget=49,
                              save_folder=save_folder,
                              name=model_name)
    policy.make()
    policy.init()

    # Train the policy
    policy.train(train_inputs=X_train, train_labels=y_train, valid_inputs=X_valid, valid_labels=y_valid, max_epochs=150, patience=50)

    # Load the test data
    X_test, y_test, _, _, _ = fetch_model_states(model=model, dataset=dataset, series=DataSeries.TEST)
    policy.test(inputs=X_test, labels=y_test)
