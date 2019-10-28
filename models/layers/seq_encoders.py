import tensorflow as tf
from typing import Tuple

from layers.basic import rnn_cell, mlp, pool_sequence


def pool_rnn_sequence(outputs: tf.Tensor, state: tf.Tensor, seq_lengths: tf.Tensor, pool_mode: str) -> tf.Tensor:
    pool_mode = pool_mode.lower()
    if pool_mode == 'hidden':
        return state
    return pool_sequence(outputs, seq_lengths, pool_mode)


def rnn_encoder(inputs: tf.Tensor,
                output_size: int,
                activation: str,
                cell_type: str,
                batch_size: int,
                seq_lengths: tf.Tensor,
                pool_mode: str,
                dropout_keep_rate: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Encodes the input sequence as a single embedding vector.

    Args:
        inputs: A B x T x D float32 tensor containing input indices
        output_size: The size of the output representation (referred to as K)
        activation: Activation function to apply within the RNN
        pool_mode: Pooling strategy of the input sequence
    Returns:
        A B x K tensor of sequence embeddings.
    """
    # Make the RNN cell
    cell = rnn_cell(cell_type=cell_type,
                    num_units=output_size,
                    activation=activation,
                    dropout_keep_rate=dropout_keep_rate,
                    name='rnn-cell',
                    dtype=tf.float32)
    # initial_state = cell.get_initial_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)
    initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

    outputs, state = tf.nn.dynamic_rnn(cell,
                                       inputs=inputs,
                                       sequence_length=seq_lengths,
                                       initial_state=initial_state,
                                       dtype=tf.float32)

    return pool_rnn_sequence(outputs, state, seq_lengths, pool_mode), outputs
