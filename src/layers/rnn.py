import tensorflow as tf
from typing import Optional, Tuple
from collections import namedtuple

from layers.basic import dense
from layers.cells.cells import MultiRNNCell
from utils.rnn_utils import get_combine_states_name, get_rnn_while_loop_name
from utils.constants import FUSION_SEED


RnnOutput = namedtuple('RnnOutput', ['outputs', 'states', 'gates'])


def fuse_states(curr_state: tf.Tensor,
                prev_state: Optional[tf.Tensor],
                state_size: int,
                mode: Optional[str],
                name: str,
                compression_fraction: Optional[float] = None,
                compression_seed: Optional[str] = None) -> tf.Tensor:
    """
    Combines the provided states using the specified strategy.

    Args:
        curr_state: A [B, D] tensor of the current state
        prev_state: An optional [B, D] tensor of previous state
        fusion_layer: Optional trainable variables for the fusion layer. Only use when mode = 'gate'
        state_size: Size (D) of the fused vector
        mode: The fusion strategy. If None is given, the strategy is an identity.
        name: Name of this layer
        compression_fraction: Compression level to apply to any trainable parameters.
        compression_seed: Seed to use for hashing function during compression.
    Returns:
        A [B, D] tensor that represents the fused state
    """
    mode = mode.lower() if mode is not None else None

    if mode is None or mode in ('identity', 'none') or prev_state is None:
        return curr_state
    elif mode == 'sum':
        return curr_state + prev_state
    elif mode in ('sum-tanh', 'sum_tanh'):
        return tf.nn.tanh(curr_state + prev_state)
    elif mode in ('avg', 'average'):
        return (curr_state + prev_state) / 2.0
    elif mode in ('max', 'max-pool', 'max_pool'):
        concat = tf.concat([tf.expand_dims(curr_state, axis=-1), tf.expand_dims(prev_state, axis=-1)], axis=-1)  # [B, D, 2]
        return tf.reduce_max(concat, axis=-1)  # [B, D]
    elif mode in ('gate', 'gate_layer', 'gate-layer'):
        concat_states = tf.concat([curr_state, prev_state], axis=-1)  # [B, 2 * D]

        # [B, D]
        update_weight, _ = dense(inputs=concat_states,
                                 units=state_size,
                                 name=name,
                                 activation='sigmoid',
                                 use_bias=True,
                                 compression_fraction=compression_fraction,
                                 compression_seed=compression_seed)

        return update_weight * curr_state + (1.0 - update_weight) * prev_state
    else:
        raise ValueError(f'Unknown fusion mode: {mode}')


def dynamic_rnn(inputs: tf.Tensor,
                cell: MultiRNNCell,
                initial_state: Optional[tf.Tensor] = None,
                previous_states: Optional[tf.TensorArray] = None,
                skip_width: Optional[int] = None,
                name: Optional[str] = None,
                should_share_weights: bool = False,
                fusion_mode: Optional[str] = None,
                compression_fraction: Optional[float] = None) -> RnnOutput:
    """
    Implementation of a recurrent neural network which allows for complex state passing.

    Args:
        inputs: A B x T x D tensor of inputs
        cell: RNN Cell to apply at each timestep
        initial_state: Optional initial state of the RNN. Defaults to a zero state.
        previous_states: Optional array of states to feed integrate into the current layer
        skip_width: Optional width of skip connections
        name: Optional name of this RNN
        should_share_weights: Whether or not to share weights for any added trainable parameters
        fusion_mode: Optional fusion mode for combining states between levels
        compression_fraction: Optional fraction for which we should compress weights
    Returns:
        A tuple of 3 tensor arrays
            (1) The outputs of each cell
            (2) The states from each cell
            (3) The gate values of each cell. This is useful for debugging / architecture comparison.
    """
    state_size = cell.state_size * cell.num_state_elements
    rnn_layers = cell.num_layers
    sequence_length = tf.shape(inputs)[1]

    states_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, dynamic_size=False, clear_after_read=False)
    outputs_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, dynamic_size=False)
    gates_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, dynamic_size=False)

    combine_layer_name = get_combine_states_name(name, should_share_weights)

    fusion_layers: List[FusionLayer] = []
    if previous_states is not None and fusion_mode.lower() == 'gate':

        # Initialize all variables before the while loop
        for i in range(rnn_layers):
            if compression_fraction is None or compression_fraction >= 1:
                state_transform = tf.get_variable(name='{0}-{1}-kernel'.format(combine_layer_name, i),
                                                  shape=(state_size * 2, state_size),
                                                  initializer=tf.initializers.glorot_uniform(),
                                                  trainable=True)
            else:
                state_transform = tf.get_variable(name='{0}-{1}-kernel'.format(combine_layer_name, i),
                                                  shape=int(compression_fraction * (state_size * state_size * 2)),
                                                  initializer=tf.initializers.glorot_uniform(),
                                                  trainable=True)

            state_transform_bias = tf.get_variable(name='{0}-{1}-bias'.format(combine_layer_name, i),
                                                   shape=(1, state_size),
                                                   initializer=tf.initializers.glorot_uniform(),
                                                   trainable=True)

    # While loop step
    def step(index, state, outputs, states, gates):
        step_inputs = tf.gather(inputs, indices=index, axis=1)  # [B, D]

        skip_inputs: Optional[tf.Tensor] = None
        if skip_width is not None:
            skip_inputs = tf.where(tf.math.less(index - skip_width, 0),
                                   x=state,
                                   y=states.read(index=index - skip_width))  # [B, D]

        # Get states
        combined_state = state
        prev_state = previous_states.read(index) if previous_states is not None else None

        # Fuse together the states
        combined_state: List[tf.Tensor] = []
        for i in range(rnn_layers):
            curr = state[i, :, :]
            prev = prev_state[i, :, :] if prev_state is not None else None

            combined = fuse_states(curr_state=curr,
                                   prev_state=prev,
                                   mode=fusion_mode,
                                   state_size=state_size,
                                   name='{0}-{1}'.format(combine_layer_name, i),
                                   compression_fraction=compression_fraction,
                                   compression_seed='{0}{1}'.format(FUSION_SEED, i))

            combined_state.append(combined)

        # Convert a stacked state into a list of states
        if not isinstance(combined_state, list):
            combined_state = [combined_state[i, :, :] for i in range(rnn_layers)]

        # Apply RNN Cell
        output, state, gates_tuple = cell(step_inputs, combined_state, skip_input=skip_inputs)

        # Save outputs
        outputs = outputs.write(index=index, value=output)
        states = states.write(index=index, value=state)

        concat_gates = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), gates_tuple),
                                 axis=1)
        gates = gates.write(index=index, value=concat_gates)

        return [index + 1, tf.stack(state), outputs, states, gates]

    def cond(index, _1, _2, _3, _4):
        return index < sequence_length

    # Index to track iteration number
    i = tf.constant(0, dtype=tf.int32)

    if initial_state is None:
        initial_state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)

    if isinstance(initial_state, list):
        initial_state = tf.stack(initial_state)

    while_loop_name = get_rnn_while_loop_name(name)
    _, _, final_outputs, final_states, final_gates = tf.while_loop(cond=cond,
                                                                   body=step,
                                                                   loop_vars=[i, initial_state, outputs_array, states_array, gates_array],
                                                                   parallel_iterations=1,
                                                                   maximum_iterations=sequence_length,
                                                                   name=while_loop_name)

    return RnnOutput(outputs=final_outputs, states=final_states, gates=final_gates)


def dropped_rnn(inputs: tf.Tensor,
                cell: MultiRNNCell,
                drop_rate: float,
                horizon: int,
                initial_state: Optional[tf.Tensor] = None,
                name: Optional[str] = None) -> Tuple[tf.TensorArray, tf.TensorArray, tf.TensorArray]:
    state_size = cell.state_size * cell.num_state_elements
    rnn_layers = cell.num_layers
    sequence_length = tf.shape(inputs)[1]
    batch_size = tf.shape(inputs)[0]

    states_array = tf.TensorArray(dtype=tf.float32, size=sequence_length + 1, dynamic_size=False, clear_after_read=False)
    outputs_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, dynamic_size=False)
    gates_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, dynamic_size=False)

    def step(index, outputs, states, gates):
        step_inputs = tf.gather(inputs, indices=index, axis=1)  # B x D

        # Randomly draw T values and store drop/keep decision as a binary vector
        step_horizon = tf.minimum(index + 1, horizon)
        rand_drop_values = tf.random.uniform(shape=(step_horizon, rnn_layers, batch_size, 1), minval=0.0, maxval=1.0)
        should_drop_mask = tf.cast(tf.math.greater(rand_drop_values, drop_rate), dtype=tf.float32)  # T x L x B x D

        # Calculate the number of remaining previous states
        num_remaining = tf.reduce_sum(should_drop_mask, axis=0)  # L x B x D

        # Read the last T values
        start = tf.maximum(0, index - horizon + 1)
        indices = tf.range(start=start, limit=index + 1, dtype=tf.int32)
        prev_states = states_array.gather(indices=indices)  # T x L x B x D

        prev_states_sum = tf.reduce_sum(should_drop_mask * prev_states, axis=0)
        aggregate_state = prev_states_sum / (num_remaining + 1e-7)

        state_per_layer = [aggregate_state[i, :, :] for i in range(rnn_layers)]
        output, state, gate_values = cell(step_inputs, state_per_layer)

        # Save outputs
        outputs = outputs.write(index=index, value=output)
        states = states.write(index=index + 1, value=state)
        gates = gates.write(index, value=gate_values)

        return [index + 1, outputs, states, gates]

    def cond(index, _1, _2, _3):
        return index < sequence_length

    if initial_state is None:
        initial_state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)

    if isinstance(initial_state, list):
        initial_state = tf.stack(initial_state)

    i = tf.constant(0, dtype=tf.int32)
    states_array = states_array.write(value=initial_state, index=0)
    _, final_outputs, final_states, final_gates = tf.while_loop(cond=cond,
                                                                body=step,
                                                                loop_vars=[i, outputs_array, states_array, gates_array],
                                                                parallel_iterations=1,
                                                                maximum_iterations=sequence_length,
                                                                name='rnn-while-loop')
    return final_outputs, final_states, final_gates
