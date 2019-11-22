import tensorflow as tf
from typing import Tuple, Dict, Optional, Any, List
from dpu_utils.tfutils import get_activation


def make_rnn_cell(cell_type: str,
                  input_units: int,
                  output_units: int,
                  activation: str,
                  dropout_keep_rate: tf.Tensor,
                  name: str,
                  num_layers: Optional[int] = None,
                  use_skip_connections: bool = False):
    if num_layers is None:
        return make_single_rnn_cell(cell_type, input_units, output_units, activation, dropout_keep_rate, name, use_skip_connections)

    return MultiRNNCell(num_layers=num_layers,
                        input_units=input_units,
                        output_units=output_units,
                        activation=activation,
                        dropout_keep_prob=dropout_keep_rate,
                        cell_type=cell_type,
                        name=name,
                        use_skip_connections=use_skip_connections)


def make_single_rnn_cell(cell_type: str,
                         input_units: int,
                         output_units: int,
                         activation: str,
                         dropout_keep_rate: tf.Tensor,
                         name: str,
                         use_skip_connections: bool = False):
    cell_type = cell_type.lower()

    if cell_type == 'gru':
        return GRU(input_units, output_units, activation, dropout_keep_rate, name, use_skip_connections)
    if cell_type == 'vanilla':
        return VanillaCell(input_units, output_units, activation, dropout_keep_rate, name, use_skip_connections)
    raise ValueError(f'Unknown cell name {cell_type}!')


class RNNCell:

    def __init__(self,
                 input_units: int,
                 output_units: int,
                 activation: str,
                 dropout_keep_prob: tf.Tensor,
                 name: str,
                 use_skip_connections: bool = False,
                 state_size: Optional[int] = None):
        """
        Initializes the RNN Cell

        Args:
            input_units: Number of dimensions of the input vectors
            output_units: Number of dimensions of the output vectors
            activation: Name of the activation function (i.e. tanh)
            dropout_keep_prob: Dropout keep rate for gate values
            name: Name of the RNN Cell
            use_skip_connections: Whether to allow skip connections through this cell
            state_size: Size of the state. Defaults to output_units
        """
        
        self.input_units = input_units
        self.output_units = output_units
        self.activation = get_activation(activation)
        self.dropout_keep_prob = dropout_keep_prob
        self.initializer = tf.initializers.glorot_uniform()
        self.state_size = output_units if state_size is None else state_size
        self.use_skip_connections = use_skip_connections
        self.name = name
        self.init_weights()

    def init_weights(self):
        """
        Initializes the trainable variables
        """
        pass

    def __call__(self, inputs: tf.Tensor,
                 state: tf.Tensor,
                 skip_input: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor]]:
        """
        Executes the RNN cell.

        Args:
            inputs: Input vectors [B, D] tensor
            state: State vectors [B, S] tensor
            skip_input: Optional skip connections, [B, S] tensor if provided
        Returns:
            A tuple of (output, state, list of gate values)
        """
        pass

    def zero_state(self, batch_size: tf.Tensor, dtype: Any) -> tf.Tensor:
        state = tf.fill(dims=[batch_size, self.state_size], value=0, name=f'{self.name}-initial-state')
        return tf.cast(state, dtype=dtype)


class MultiRNNCell(RNNCell):

    def __init__(self,
                 num_layers: int,
                 input_units: int,
                 output_units: int,
                 activation: str,
                 dropout_keep_prob: tf.Tensor,
                 name: str,
                 cell_type: str,
                 use_skip_connections: bool = False,
                 state_size: Optional[int] = None):
        assert num_layers >= 1, 'Must provide at least one layer'
        super().__init__(input_units, output_units, activation, dropout_keep_prob, name, use_skip_connections, state_size)
        self.num_layers = num_layers
        
        self.cells : List[RNNCell] = []
        for i in range(num_layers):
            cell = make_single_rnn_cell(cell_type=cell_type,
                                        input_units=input_units if i == 0 else output_units,
                                        output_units=output_units,
                                        activation=activation,
                                        dropout_keep_rate=dropout_keep_prob,
                                        name=f'{name}-cell-{i}',
                                        use_skip_connections=use_skip_connections)
            self.cells.append(cell)

    def zero_state(self, batch_size: tf.Tensor, dtype: Any) -> List[tf.Tensor]:
        return [cell.zero_state(batch_size, dtype) for cell in self.cells]

    def __call__(self, inputs: tf.Tensor,
                 state: List[tf.Tensor],
                 skip_input: Optional[List[tf.Tensor]] = None) -> Tuple[tf.Tensor, List[tf.Tensor], List[List[tf.Tensor]]]: 
        assert len(self.cells) == len(state), 'The number of states must be equal to the number of cells'

        cell_gates: List[List[tf.Tensor]] = []
        cell_states: List[tf.Tensor] = []
        cell_outputs: List[tf.Tensor] = [inputs]

        for i, (cell, state) in enumerate(zip(self.cells, state)):
            skip_connection = skip_input[i] if skip_input is not None else None

            cell_output, cell_state, cell_gate = cell(inputs=cell_outputs[-1],
                                                      state=state,
                                                      skip_input=skip_connection)
            cell_outputs.append(cell_output)
            cell_states.append(cell_state)
            cell_gates.append(cell_gate)

        final_output = cell_outputs[-1]
        return final_output, cell_states, cell_gates


class GRU(RNNCell):

    def init_weights(self):
        self.W_update = tf.Variable(initial_value=self.initializer(shape=[self.state_size, self.output_units]),
                                    trainable=True,
                                    name=f'{self.name}-W-update')
        self.U_update = tf.Variable(initial_value=self.initializer(shape=[self.input_units, self.output_units]),
                                    trainable=True,
                                    name=f'{self.name}-U-update')
        self.b_update = tf.Variable(initial_value=self.initializer(shape=[1, self.output_units]),
                                    trainable=True,
                                    name=f'{self.name}-b-update')

        self.W_reset = tf.Variable(initial_value=self.initializer(shape=[self.state_size, self.output_units]),
                                    trainable=True,
                                    name=f'{self.name}-W-reset')
        self.U_reset = tf.Variable(initial_value=self.initializer(shape=[self.input_units, self.output_units]),
                                    trainable=True,
                                    name=f'{self.name}-U-reset')
        self.b_reset = tf.Variable(initial_value=self.initializer(shape=[1, self.output_units]),
                                   trainable=True,
                                   name=f'{self.name}-b-reset')

        self.W = tf.Variable(initial_value=self.initializer(shape=[self.state_size, self.output_units]),
                             trainable=True,
                             name=f'{self.name}-W')
        self.U = tf.Variable(initial_value=self.initializer(shape=[self.input_units, self.output_units]),
                             trainable=True,
                             name=f'{self.name}-U')
        self.b = tf.Variable(initial_value=self.initializer(shape=[1, self.output_units]),
                             trainable=True,
                             name=f'{self.name}-b')

        if self.use_skip_connections:
            self.R_update = tf.Variable(initial_value=self.initializer(shape=[self.state_size, self.output_units]),
                                        trainable=True,
                                        name=f'{self.name}-R-update')
            self.R_reset = tf.Variable(initial_value=self.initializer(shape=[self.state_size, self.output_units]),
                                    trainable=True,
                                    name=f'{self.name}-R-update')
            self.R = tf.Variable(initial_value=self.initializer(shape=[self.state_size, self.output_units]),
                                 trainable=True,
                                 name=f'{self.name}-R')

    def __call__(self, inputs: tf.Tensor,
                 state: tf.Tensor,
                 skip_input: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor]]:
        assert not self.use_skip_connections or skip_input is not None, 'Must provide a skip input when using skip connections'

        update_vector = tf.matmul(state, self.W_update) + tf.matmul(inputs, self.U_update) + self.b_update
        reset_vector = tf.matmul(state, self.W_reset) + tf.matmul(inputs, self.U_reset) + self.b_reset
        candidate_vector = tf.matmul(state, self.W) + tf.matmul(inputs, self.U) + self.b

        if self.use_skip_connections:
            update_vector += tf.matmul(skip_input, self.R_update)
            reset_gate += tf.matmul(skip_input, self.R_reset)
            candidate_vector += tf.matmul(skip_input, self.R)

        update_gate = tf.math.sigmoid(update_vector)
        reset_gate = tf.math.sigmoid(reset_vector)

        update_with_dropout = tf.nn.dropout(update_gate, keep_prob=self.dropout_keep_prob)
        reset_with_dropout = tf.nn.dropout(reset_gate, keep_prob=self.dropout_keep_prob)

        candidate_state = self.activation(candidate_vector)
        next_state = update_with_dropout * state + (1.0 - update_with_dropout) * candidate_state

        return next_state, next_state, [update_gate, reset_gate]


class VanillaCell(RNNCell):

    def init_weights(self):
        self.W = tf.Variable(initial_value=self.initializer(shape=[self.state_size, self.output_units]),
                             trainable=True,
                             name=f'{self.name}-W')
        self.U = tf.Variable(initial_value=self.initializer(shape=[self.input_units, self.output_units]),
                             trainable=True,
                             name=f'{self.name}-U')
        self.b = tf.Variable(initial_value=self.initializer(shape=[1, self.output_units]),
                             trainable=True,
                             name=f'{self.name}-b')

        if self.use_skip_connections:
            self.R = tf.Variable(initial_value=self.initializer(shape=[self.input_units, self.output_units]),
                                 trainable=True,
                                 name=f'{self.name}-R')

    def __call__(self, inputs: tf.Tensor,
                 state: tf.Tensor,
                 skip_input: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor]]:
        assert not self.use_skip_connections or skip_input is None, 'Must provide a skip input when using skip connections'

        candidate_vector = tf.matmul(state, self.W) + tf.matmul(inputs, self.U) + self.b

        if self.use_skip_connections:
            candidate_vector += tf.matmul(skip_input, self.R)

        next_state = self.activation(candidate_vector)
        next_state = tf.nn.dropout(next_state, keep_prob=self.dropout_keep_prob)
        return next_state, next_state, [candidate_vector]
