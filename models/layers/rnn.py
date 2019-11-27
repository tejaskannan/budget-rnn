import tensorflow as tf
from typing import Optional, Tuple

from layers.cells.cells import MultiRNNCell


def dynamic_rnn(inputs: tf.Tensor,
                cell: MultiRNNCell,
                initial_state: Optional[tf.Tensor] = None,
                previous_states: Optional[tf.TensorArray] = None,
                skip_width: Optional[int] = None,
                name: Optional[str] = None) -> Tuple[tf.TensorArray, tf.TensorArray, tf.TensorArray]:
    """
    Implementation of a recurrent neural network which allows for complex state passing.

    Args:
        inputs: A B x T x D tensor of inputs
        cell: RNN Cell to apply at each timestep
        initial_state: Optional initial state of the RNN. Defaults to a zero state.
        previous_states: Optional array of states to feed integrate into the current layer
        skip_width: Optional width of skip connections
        name: Optional name of this RNN
    Returns:
        A tuple of 3 tensor arrays
            (1) The outputs of each cell
            (2) The states from each cell
            (3) The gate values of each cell. This is useful for debugging / architecture comparison.
    """
    state_size = cell.state_size
    rnn_layers = cell.num_layers
    sequence_length = tf.shape(inputs)[1]

    states_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, dynamic_size=False, clear_after_read=False)
    outputs_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, dynamic_size=False)
    gates_array = tf.TensorArray(dtype=tf.float32, size=sequence_length, dynamic_size=False)

    if previous_states is not None:
        combine_layer_name = 'combine-states' if name is None else f'{name}-combine-states'

        prev_state_transform_layers: List[tf.layers.Dense] = []
        curr_state_transform_layers: List[tf.layers.Dense] = []
        state_transform_bias_layers: List[tf.Variable] = []

        for i in range(rnn_layers):
            prev_state_transform = tf.layers.Dense(units=state_size,
                                                   activation=None,
                                                   use_bias=False,
                                                   kernel_initializer=tf.initializers.glorot_uniform(),
                                                   name=combine_layer_name + f'-prev-{i}')
            curr_state_transform = tf.layers.Dense(units=state_size,
                                                   activation=None,
                                                   use_bias=False,
                                                   kernel_initializer=tf.initializers.glorot_uniform(),
                                                   name=combine_layer_name + f'-curr-{i}')
            state_transform_bias = tf.Variable(initial_value=tf.random.normal(shape=(1, state_size)),
                                               trainable=True,
                                               name=combine_layer_name + f'-bias-{i}')

            prev_state_transform_layers.append(prev_state_transform)
            curr_state_transform_layers.append(curr_state_transform)
            state_transform_bias_layers.append(state_transform_bias)

    # While loop step
    def step(index, state, outputs, states, gates):
        step_inputs = tf.gather(inputs, indices=index, axis=1)  # B x D

        skip_inputs: Optional[tf.Tensor] = None
        if skip_width is not None:
            skip_inputs = tf.where(tf.math.less_equal(index - skip_width, -1),
                                   x=tf.zeros_like(step_inputs),
                                   y=tf.gather(inputs, indices=index - skip_width, axis=1))  # B x D

        combined_state = state
        if previous_states is not None:
            prev_state = previous_states.read(index)

            # This sequence of operations mirrors a GRU update gate.
            prev_transform = prev_state_transform(prev_state)

            combined_state: List[tf.Tensor] = []
            for i in range(rnn_layers):
                curr = state[i, :, :]
                curr_transform = curr_state_transform_layers[i](curr)

                prev = prev_state[i, :, :]
                prev_transform = prev_state_transform_layers[i](prev)

                update_weight = tf.math.sigmoid(prev_transform + curr_transform + state_transform_bias_layers[i])

                combined = update_weight * curr + (1.0 - update_weight) * prev
                combined_state.append(combined)

        if not isinstance(combined_state, list):
            combined_state = [combined_state[i, :, :] for i in range(rnn_layers)]

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

    i = tf.constant(0, dtype=tf.int32)

    if initial_state is None:
        initial_state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)

    if isinstance(initial_state, list):
        initial_state = tf.stack(initial_state)

    _, _, final_outputs, final_states, final_gates = tf.while_loop(cond=cond,
                                                                   body=step,
                                                                   loop_vars=[i, initial_state, outputs_array, states_array, gates_array],
                                                                   parallel_iterations=1,
                                                                   maximum_iterations=sequence_length,
                                                                   name='rnn-while-loop')
    return final_outputs, final_states, final_gates
