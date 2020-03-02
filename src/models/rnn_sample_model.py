import tensorflow as tf

from layers.basic import rnn_cell, mlp
from layers.cells.cells import make_rnn_cell, MultiRNNCell
from layers.rnn import dynamic_rnn
from layers.output_layers import compute_regression_output, compute_binary_classification_output, OutputType
from layers.output_layers import ClassificationOutput, RegressionOutput
from utils.rnn_utils import *
from utils.tfutils import pool_rnn_outputs
from .rnn_model import RNNModel


class RNNSampleModel(RNNModel):


    def make_rnn_model(self, is_train: bool):
        outputs: List[tf.Tensor] = []  # Holds outputs from each level
        states_list: List[tf.TensorArray] = []  # Holds states from each level

        num_input_features = self.metadata['num_input_features']
        num_output_features = self.metadata['num_output_features']

        for i in range(self.num_sequences):
            # Get relevant variable names
            input_name = get_input_name(i)
            cell_name = get_cell_level_name(i, self.hypers.model_params['share_cell_weights'])
            rnn_level_name = get_rnn_level_name(i)
            output_layer_name = get_output_layer_name(i)
            logits_name = get_logits_name(i)
            prediction_name = get_prediction_name(i)
            loss_name = get_loss_name(i)
            gate_name = get_gates_name(i)
            state_name = get_states_name(i)
            accuracy_name = get_accuracy_name(i)

            # Create Recurrent Cell
            cell = make_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                 input_units=num_input_features,
                                 output_units=self.hypers.model_params['state_size'],
                                 activation=self.hypers.model_params['rnn_activation'],
                                 dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                                 num_layers=self.hypers.model_params['rnn_layers'],
                                 name=cell_name)

            # Execute the RNN
            prev_states = states_list[i-1] if i > 0 else None
            rnn_outputs, states, gates = dynamic_rnn(inputs=self._placeholders[input_name],
                                                     cell=cell,
                                                     previous_states=prev_states,
                                                     name=rnn_level_name)

            # Appends states to list
            last_index = tf.shape(self._placeholders[input_name])[1] - 1
            states_list.append(states)

            # Compute Recurrent Output
            final_output = rnn_outputs.read(last_index)  # [B, D]
            final_state = states.read(last_index)  # [B, D]
            rnn_output = pool_rnn_outputs(rnn_outputs, final_output, pool_mode=self.hypers.model_params['pool_mode'])

            # B x D'
            output = mlp(inputs=rnn_output,
                         output_size=num_output_features,
                         hidden_sizes=self.hypers.model_params.get('output_hidden_units'),
                         activations=self.hypers.model_params['output_hidden_activation'],
                         dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                         name=output_layer_name)

            if self.output_type == OutputType.CLASSIFICATION:
                classification_output = compute_binary_classification_output(model_output=output,
                                                                             labels=self._placeholders['output'],
                                                                             false_pos_weight=self.hypers.model_params['pos_weights'][i],
                                                                             false_neg_weight=self.hypers.model_params['neg_weights'][i])
                self._ops[logits_name] = classification_output.logits,
                self._ops[prediction_name] = classification_output.predictions
                self._ops[loss_name] = classification_output.loss
                self._ops[accuracy_name] = classification_output.accuracy
            else:
                regression_output = compute_regression_output(model_output=output, expected_otuput=self._placeholders['output'])
                self._ops[prediction_name] = regression_output.predictions
                self._ops[loss_name] = regression_output.loss

            self._ops[gate_name] = gates.stack()  # [B, T, M, D]
            self._ops[state_name] = states.stack()  # [B, T, D]

            outputs.append(output)

        combined_outputs = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), outputs), axis=1)
        self._ops[ALL_PREDICTIONS_NAME] = combined_outputs  # [B, N, D']
        self._loss_ops = self._make_loss_ops(use_previous_layers=True)



