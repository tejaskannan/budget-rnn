import tensorflow as tf
import collections

from tensorflow.python.ops.rnn_cell_impl import assert_like_rnncell
from utils.constants import SMALL_NUMBER, BIG_NUMBER


def unnormalized_luong_attention(*args):
    return tf.contrib.seq2seq.LuongAttention(*args, probability_fn=lambda scores: scores)


def batch_unsrt_segment_max(inputs: tf.Tensor,
                            segment_ids: tf.Tensor,
                            num_segments: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(inputs)[0]
    batch_indices = tf.range(batch_size)  # B

    segment_ids_per_batch = segment_ids + num_segments * tf.expand_dims(batch_indices, axis=-1)  # B

    seg_maxes = tf.unsorted_segment_max(inputs, segment_ids_per_batch, num_segments * batch_size)
    return tf.reshape(seg_maxes, shape=[-1, num_segments])


def batch_unsrt_segment_sum(inputs: tf.Tensor,
                            segment_ids: tf.Tensor,
                            num_segments: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(inputs)[0]
    batch_indices = tf.range(batch_size)  # B

    segment_ids_per_batch = segment_ids + num_segments * tf.expand_dims(batch_indices, axis=-1)  # B

    seg_sums = tf.unsorted_segment_sum(inputs, segment_ids_per_batch, num_segments * batch_size)
    return tf.reshape(seg_sums, shape=[-1, num_segments])


def batch_unsrt_segment_logsumexp(inputs: tf.Tensor,
                                  segment_ids: tf.Tensor,
                                  num_segments: tf.Tensor) -> tf.Tensor:
    # Compute the maximum of each segment
    #max_values = batch_unsrt_segment_max(inputs, segment_ids, num_segments)  # B x V'
    #data_max = tf.gather(max_values, segment_ids)  # B x T

    #data = inputs - data_max
    data_exp = tf.exp(inputs)
    data_sum = batch_unsrt_segment_sum(inputs=data_exp,
                                       segment_ids=segment_ids,
                                       num_segments=num_segments)
    data_log = tf.log(data_sum + SMALL_NUMBER)  # B x V'
    return data_log


class CopyingWrapperState(
        collections.namedtuple("AttentionWrapperState",
                               ("cell_state", "alignments", "time", "copying_state"))):
    """`namedtuple` storing the state of a `CopyingWrapper`."""

    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tf.contrib.framework.with_same_shape(old, new)
            return new

        return tf.contrib.framework.nest.map_structure(
                with_same_shape,
                self,
                super(CopyingWrapperState, self)._replace(**kwargs))


class CopyingWrapper(tf.nn.rnn_cell.RNNCell):
    """Wraps another `RNNCell` with copying.
    """
    def __init__(self,
                 cell,
                 copying_mechanism,
                 memory_out_ids,
                 extended_vocab_size,
                 output_layer=None,
                 initial_cell_state=None,
                 name=None):

        super().__init__(name=name)
        if not isinstance(copying_mechanism, tf.contrib.seq2seq.AttentionMechanism):
            raise TypeError(
                "copying_mechanism must be an AttentionMechanism saw type: %s"
                % type(copying_mechanism).__name__)

        self._cell = cell
        self._copying_mechanism = copying_mechanism
        self._memory_out_ids = memory_out_ids
        self._extended_vocab_size = extended_vocab_size
        self._output_layer = output_layer
        with tf.name_scope(name, "CopyingWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = tf.contrib.framework.flatten(initial_cell_state)[-1]
                state_batch_size = (
                    final_state_tensor.shape[0].value
                    or tf.shape(final_state_tensor)[0])
                error_message = (
                    "When constructing CopyingWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and initial_cell_state.  Are you using "
                    "the BeamSearchDecoder?  You may need to tile your initial state "
                    "via the tf.contrib.seq2seq.tile_batch function with argument "
                    "multiple=beam_width.")
                with tf.control_dependencies(
                      self._batch_size_checks(state_batch_size, error_message)):
                    self._initial_cell_state = tf.contrib.framework.nest.map_structure(
                        lambda s: tf.identity(s, name="check_initial_cell_state"),
                        initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        return tf.assert_equal(batch_size,
                               self._copying_mechanism.batch_size,
                               message=error_message)

    @property
    def output_size(self):
        return tf.expand_dims(tf.cast(self._extended_vocab_size, tf.int32), -1)

    @property
    def state_size(self):
        return CopyingWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            alignments=self._copying_mechanism.alignments_size,
            copying_state=self._copying_mechanism.state_size)

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)

        return CopyingWrapperState(
            cell_state=cell_state,
            time=tf.zeros([], dtype=tf.int32),
            alignments=self._copying_mechanism.initial_alignments(batch_size, dtype),
            copying_state=self._copying_mechanism.initial_state(batch_size, dtype))

    def _get_state(self, cell_state, lstm_output_only=False):
        # multi-layer RNN
        if isinstance(cell_state, tuple) and not isinstance(cell_state, tf.nn.rnn_cell.LSTMStateTuple):
            cell_state = cell_state[-1]
        # LSTM
        if isinstance(cell_state, tf.nn.rnn_cell.LSTMStateTuple):
            cell_state = (tf.concat([cell_state.h, cell_state.c], axis=-1)
                          if not lstm_output_only else cell_state.h)
        return cell_state

    def _extend_logits(self, logits, extended_vocab_size):
        batch_size = tf.shape(logits)[0]
        pad_size = extended_vocab_size - tf.shape(logits)[1]
        padding = tf.fill(tf.stack((batch_size, pad_size)), value=-BIG_NUMBER)
        return tf.concat([logits, padding], axis=1)

    def call(self, inputs, state):
        """
        B: batch size
        V: vocab size
        V': extended vocab size
        D: state dimension size
        T: input sequence length
        """
        if not isinstance(state, CopyingWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead."  % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(inputs, cell_state)  # B x D, B x D

        if isinstance(next_cell_state, tf.contrib.seq2seq.AttentionWrapperState):
            raw_cell_state = next_cell_state.cell_state
        else:
            raw_cell_state = next_cell_state

        if self._output_layer is not None:
            cell_output = self._output_layer(cell_output)

        # extract logits for output vocabulary based on the base decoder
        # and extend them so that we have logits for the full extended vocabulary
        # logits for the extended tokens will be set to (pratically) minus infinity
        base_logits = cell_output
        vocab_size = tf.shape(base_logits)[1]
        base_logits = self._extend_logits(base_logits, self._extended_vocab_size)  # B x V'

        # calculate logits for words in the input based on copying attention
        # copying_alignments is a B x T tensor, copying_state is a B x T tensor
        copying_alignments, copying_state = self._copying_mechanism(
            self._get_state(raw_cell_state, lstm_output_only=True), state.copying_state)

        # creates a logit distribution for the full extended vocabulary, 
        # by adding in probability space the logits for input elements corresponding to 
        # the same token. Output vocabulary tokens that don't appear in the input sequence
        # will be set to (pratically) minus infinity
        copying_logits = batch_unsrt_segment_logsumexp(
            copying_alignments,
            segment_ids=self._memory_out_ids,
            num_segments=self._extended_vocab_size)

        # create new distribution from these two logits distributions
        # note that this is slightly less sound since it assumes the model
        # is able to learn to scale between both distributions instead of 
        # having an explicit gating for merging probabilites
        # however it allows us to work with logits instead of logprobs
        # and adds more capacitity to the model
        concat_logits = tf.stack(
            [base_logits, copying_logits],
            axis=-1)  # B x V' x 2
 
        final_logits = tf.reduce_logsumexp(concat_logits, axis=-1)  # B x V'

        # Mask out all logits after the maximum vocab term (vocab is padded since
        # extensions can be variable based on the number of OOV input tokens)
        batch_size = tf.shape(final_logits)[0]
        vocab_ids = tf.expand_dims(tf.range(tf.shape(final_logits)[1]), axis=0)  # 1 x V'
        vocab_ids = tf.cast(tf.tile(vocab_ids, multiples=(batch_size, 1)), dtype=tf.float32)
 
        max_vocab_id = tf.reduce_max(self._memory_out_ids, axis=1, keepdims=True)  # B x 1
        max_vocab_id = tf.cast(max_vocab_id, dtype=tf.float32)

        vocab_size_tensor = tf.fill(dims=(batch_size, 1), value=float(vocab_size))
        max_vocab_id = tf.maximum(vocab_size_tensor, max_vocab_id)
        vocab_mask = -BIG_NUMBER * (1.0 - tf.cast(vocab_ids <= max_vocab_id, dtype=tf.float32))

        final_logits = final_logits + vocab_mask

        next_state = CopyingWrapperState(
            cell_state=next_cell_state,
            time=state.time + 1,
            alignments=copying_alignments,
            copying_state=copying_state)

        return final_logits, next_state

    # TODO AND WARNING: this is a hotfix so that copying works for trained models.
    # this will be removed in the future
    def _set_scope(self, scope=None):
        if self._scope is None:
            if scope is None:
                self._scope = tf.get_variable_scope()
                return
