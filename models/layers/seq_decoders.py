import tensorflow as tf
from typing import Optional, Tuple

from layers.cells.copy_attention_wrapper import CopyingWrapper
from layers.basic import rnn_cell


def rnn_decoder(embedding: tf.Tensor,
                seq_lengths: tf.Tensor,
                start_tokens: tf.Tensor,
                end_token: int,
                input_sequence: Optional[tf.Tensor],
                vocab_size: int,
                beam_width: int,
                cell_type: str,
                activation: str,
                state_size: int,
                batch_size: int,
                dropout_keep_rate: tf.Tensor,
                initial_state: tf.Tensor,
                is_train: bool,
                maximum_iterations: int = 100) -> Tuple[tf.Tensor, tf.Tensor]:
    
    # Create RNN cell and output layer (which produces log probabilities)
    cell = rnn_cell(cell_type=cell_type,
                    num_units=state_size,
                    activation=activation,
                    dropout_keep_rate=dropout_keep_rate,
                    name='rnn-decoder-cell',
                    dtype=tf.float32)
    projection_layer = tf.layers.Dense(units=vocab_size,
                                       activation=None,
                                       use_bias=False,
                                       kernel_initializer=tf.initializers.glorot_uniform(),
                                       name='decoder-projection-layer')
    
    # Tensorflow will complain about NoneType dimensions in the initial state,
    # so set the shape explicitly here
    initial_state.set_shape([batch_size, initial_state.get_shape()[1]])

    if is_train:
        if input_sequence is None:
            raise ValueError('The input sequence cannot be None in training mode.')

        input_embeddings = tf.nn.embedding_lookup(embedding, input_sequence)
        input_embeddings.set_shape([batch_size, input_embeddings.get_shape()[1], input_embeddings.get_shape()[2]])

        helper = tf.contrib.seq2seq.TrainingHelper(input_embeddings, seq_lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state=initial_state, output_layer=projection_layer)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        ids, log_probs = outputs.sample_id, outputs.rnn_output
        return ids, log_probs
    else:
        # Tile initial state across all beams
        tiled_initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier=beam_width)

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell,
                                                       embedding=embedding,
                                                       start_tokens=start_tokens,
                                                       end_token=end_token,
                                                       initial_state=tiled_initial_state,
                                                       beam_width=beam_width,
                                                       output_layer=projection_layer)
        outputs, beam_state, length = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        ids, log_probs = outputs.predicted_ids[:, :, 0], beam_state.log_probs[:, 0]
 
        return ids, log_probs


def rnn_decoder_with_attention(memory: tf.Tensor,
                               memory_seq_lengths: tf.Tensor,
                               embedding: tf.Tensor,
                               seq_lengths: tf.Tensor,
                               start_tokens: tf.Tensor,
                               end_token: int,
                               input_sequence: Optional[tf.Tensor],
                               vocab_size: int,
                               beam_width: int,
                               cell_type: str,
                               activation: str,
                               state_size: int,
                               batch_size: int,
                               dropout_keep_rate: tf.Tensor,
                               initial_state: tf.Tensor,
                               is_train: bool,
                               maximum_iterations: int = 100) -> Tuple[tf.Tensor, tf.Tensor]:
    # Create RNN cell and output layer (which produces log probabilities)
    cell = rnn_cell(cell_type=cell_type,
                    num_units=state_size,
                    activation=activation,
                    dropout_keep_rate=dropout_keep_rate,
                    name='rnn-decoder-cell',
                    dtype=tf.float32)
    projection_layer = tf.layers.Dense(units=vocab_size,
                                       activation=None,
                                       use_bias=False,
                                       kernel_initializer=tf.initializers.glorot_uniform(),
                                       name='decoder-projection-layer')
    
    # Tensorflow will complain about NoneType dimensions in the initial state,
    # so set the shape explicitly here
    initial_state.set_shape([batch_size, initial_state.get_shape()[1]])

    if is_train:
        if input_sequence is None:
            raise ValueError('The input sequence cannot be None in training mode.')
        
        # Wrap the RNN cell with the attention layer
        attention_layer = tf.contrib.seq2seq.BahdanauAttention(num_units=state_size,
                                                               memory=memory,
                                                               memory_sequence_length=memory_seq_lengths,
                                                               dtype=tf.float32)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_layer, attention_layer_size=state_size)

        # Set initial state for the Attention-Wrapped cell
        cell_initial_state = attention_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        initial_state = cell_initial_state.clone(cell_state=initial_state)

        input_embeddings = tf.nn.embedding_lookup(embedding, input_sequence)
        input_embeddings.set_shape([batch_size, input_embeddings.get_shape()[1], input_embeddings.get_shape()[2]])

        helper = tf.contrib.seq2seq.TrainingHelper(input_embeddings, seq_lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, initial_state=initial_state, output_layer=projection_layer)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        ids, log_probs = outputs.sample_id, outputs.rnn_output
        return ids, log_probs
    else:
        # Tile initial state and memory across all beams
        tiled_initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier=beam_width)
        tiled_memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
        tiled_memory_seq_lengths = tf.contrib.seq2seq.tile_batch(memory_seq_lengths, multiplier=beam_width)

        # Wrap the RNN cell with the attention layer
        attention_layer = tf.contrib.seq2seq.BahdanauAttention(num_units=state_size,
                                                               memory=tiled_memory,
                                                               memory_sequence_length=tiled_memory_seq_lengths,
                                                               dtype=tf.float32)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_layer, attention_layer_size=state_size)

        # Set the initial state for the Attention-Wrapped cell
        cell_initial_state = attention_cell.zero_state(batch_size=batch_size * beam_width, dtype=tf.float32)
        initial_state = cell_initial_state.clone(cell_state=tiled_initial_state)

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=attention_cell,
                                                       embedding=embedding,
                                                       start_tokens=start_tokens,
                                                       end_token=end_token,
                                                       initial_state=initial_state,
                                                       beam_width=beam_width,
                                                       output_layer=projection_layer)
        outputs, beam_state, length = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        ids, log_probs = outputs.predicted_ids[:, :, 0], beam_state.log_probs[:, 0]
 
        return ids, log_probs


def rnn_decoder_with_copying(memory: tf.Tensor,
                             memory_seq_lengths: tf.Tensor,
                             memory_out_ids: tf.Tensor,
                             embedding: tf.Tensor,
                             seq_lengths: tf.Tensor,
                             start_tokens: tf.Tensor,
                             end_token: int,
                             input_sequence: Optional[tf.Tensor],
                             vocab_size: int,
                             extended_vocab_size: int,
                             beam_width: int,
                             cell_type: str,
                             activation: str,
                             state_size: int,
                             batch_size: int,
                             dropout_keep_rate: tf.Tensor,
                             initial_state: tf.Tensor,
                             is_train: bool,
                             unk_id: int = 0,
                             maximum_iterations: int = 100) -> Tuple[tf.Tensor, tf.Tensor]:
    # Create RNN cell and output layer (which produces log probabilities)
    cell = rnn_cell(cell_type=cell_type,
                    num_units=state_size,
                    activation=activation,
                    dropout_keep_rate=dropout_keep_rate,
                    name='rnn-decoder-cell',
                    dtype=tf.float32)
    projection_layer = tf.layers.Dense(units=vocab_size,
                                       activation=None,
                                       use_bias=False,
                                       kernel_initializer=tf.initializers.glorot_uniform(),
                                       name='decoder-projection-layer')

    # Tensorflow will complain about NoneType dimensions in the initial state,
    # so set the shape explicitly here
    initial_state.set_shape([batch_size, initial_state.get_shape()[1]])

    unk_embedding = embedding[unk_id, :]  # D
    embedding_padding = tf.tile(tf.expand_dims(unk_embedding, axis=0), multiples=(extended_vocab_size - vocab_size, 1))
    # embedding_padding = tf.fill(dims=[extended_vocab_size - vocab_size, tf.shape(embedding)[1]], value=0.0)
    embedding = tf.concat([embedding, embedding_padding], axis=0)  # V' x D

    if is_train:
        if input_sequence is None:
            raise ValueError('The input sequence cannot be None in training mode.')
        
        # Wrap the RNN cell with the attention layer
        attention_layer = tf.contrib.seq2seq.BahdanauAttention(num_units=state_size,
                                                               memory=memory,
                                                               memory_sequence_length=memory_seq_lengths,
                                                               dtype=tf.float32) 
        copying_cell = CopyingWrapper(cell=cell,
                                      copying_mechanism=attention_layer,
                                      memory_out_ids=memory_out_ids,
                                      extended_vocab_size=extended_vocab_size,
                                      output_layer=projection_layer)

        # Set initial state for the Copying cell
        cell_initial_state = copying_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        initial_state = cell_initial_state.clone(cell_state=initial_state)

        input_embeddings = tf.nn.embedding_lookup(embedding, input_sequence)
        input_embeddings.set_shape([batch_size, input_embeddings.get_shape()[1], input_embeddings.get_shape()[2]])

        helper = tf.contrib.seq2seq.TrainingHelper(input_embeddings, seq_lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(copying_cell, helper, initial_state=initial_state)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        ids, log_probs = outputs.sample_id, outputs.rnn_output
        return ids, log_probs
    else:
        # Tile initial state and memory across all beams
        tiled_initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier=beam_width)
        tiled_memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
        tiled_memory_seq_lengths = tf.contrib.seq2seq.tile_batch(memory_seq_lengths, multiplier=beam_width)
        tiled_memory_ids = tf.contrib.seq2seq.tile_batch(memory_out_ids, multiplier=beam_width)

        # Wrap the RNN cell with the attention layer
        attention_layer = tf.contrib.seq2seq.BahdanauAttention(num_units=state_size,
                                                               memory=tiled_memory,
                                                               memory_sequence_length=tiled_memory_seq_lengths,
                                                               dtype=tf.float32)

        copying_cell = CopyingWrapper(cell=cell,
                                      copying_mechanism=attention_layer,
                                      memory_out_ids=tiled_memory_ids,
                                      extended_vocab_size=extended_vocab_size,
                                      output_layer=projection_layer)

        # Set the initial state for the Attention-Wrapped cell
        cell_initial_state = copying_cell.zero_state(batch_size=batch_size * beam_width, dtype=tf.float32)
        initial_state = cell_initial_state.clone(cell_state=tiled_initial_state)

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=copying_cell,
                                                       embedding=embedding,
                                                       start_tokens=start_tokens,
                                                       end_token=end_token,
                                                       initial_state=initial_state,
                                                       beam_width=beam_width)
        outputs, beam_state, length = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        ids, log_probs = outputs.predicted_ids[:, :, 0], beam_state.log_probs[:, 0]
 
        return ids, log_probs
