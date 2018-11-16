#!/usr/bin/env python
# -*- coding: utf-8 -*- 

#
# Copyright 2018 Guenter Bartsch
# Copyright 2018 Keith Ito
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# tensorflow tacotron implementation
#
# heavily based on
# * https://github.com/keithito/tacotron
# * https://github.com/librosa/librosa
#

import os
import logging
import json
import codecs
import random

import numpy             as np
import tensorflow        as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.contrib.rnn     import GRUCell, RNNCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import AttentionWrapper, BahdanauAttention, Helper, BasicDecoder
from time                       import time
from nltools.tokenizer          import tokenize
from .                          import DSFN_X, DSFN_XL, DSFN_YS, DSFN_YM, DSFN_YL, VOICE_PATH, CHECKPOINT_FN, WAV_FN, SPEC_FN, ALIGN_FN, cleanup_text

import audio

DEBUG_LIMIT        =   0
# DEBUG_LIMIT        =   2 # debug only
DEFAULT_NUM_EPOCHS = 300

def _go_frames(batch_size, output_dim):
    '''Returns all-zero <GO> frames for a given batch size and output dimension'''
    return tf.tile([[0.0]], [batch_size, output_dim])

# Adapted from tf.contrib.seq2seq.GreedyEmbeddingHelper
class TacoTestHelper(Helper):
    def __init__(self, batch_size, output_dim, r):
        with tf.name_scope('TacoTestHelper'):
            self._batch_size = batch_size
            self._output_dim = output_dim
            self._end_token = tf.tile([0.0], [output_dim * r])
  
    @property
    def batch_size(self):
        return self._batch_size
  
    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])
  
    @property
    def sample_ids_dtype(self):
        return np.int32
  
    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))
  
    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them
  
    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        '''Stop on EOS. Otherwise, pass the last output as the next input and pass through state.'''
        with tf.name_scope('TacoTestHelper'):
            finished = tf.reduce_all(tf.equal(outputs, self._end_token), axis=1)
            # Feed last output frame as next input. outputs is [N, output_dim * r]
            next_inputs = outputs[:, -self._output_dim:]
            return (finished, next_inputs, state)

class TacoTrainingHelper(Helper):
    def __init__(self, inputs, targets, output_dim, r, lengths):
        # inputs is [N, T_in], targets is [N, T_out, D]
  
  
        with tf.name_scope('TacoTrainingHelper'):
            self._batch_size = tf.shape(inputs)[0]
            self._output_dim = output_dim
  
            # Feed every r-th target frame as input
            self._targets = targets[:, r-1::r, :]
  
            # Use full length for every target because we don't want to mask the padding frames
            num_steps = tf.shape(self._targets)[1]
            self._lengths = tf.tile([num_steps], [self._batch_size])
            # FIXME self._lengths = lengths
  
    @property
    def batch_size(self):
        return self._batch_size
  
    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])
  
    @property
    def sample_ids_dtype(self):
        return np.int32
  
    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))
  
    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them
  
    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name or 'TacoTrainingHelper'):
            finished = (time + 1 >= self._lengths)
            next_inputs = self._targets[:, time, :]
            return (finished, next_inputs, state)

class LocationSensitiveAttention(BahdanauAttention):
    '''Implements Location Sensitive Attention from:
    Chorowski, Jan et al. 'Attention-Based Models for Speech Recognition'
    https://arxiv.org/abs/1506.07503
    '''
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 filters=20,
                 kernel_size=7,
                 name='LocationSensitiveAttention'):
        '''Construct the Attention mechanism. See superclass for argument details.'''
        super(LocationSensitiveAttention, self).__init__( num_units,
                                                          memory,
                                                          memory_sequence_length=memory_sequence_length,
                                                          name=name)
        self.location_conv = tf.layers.Conv1D( filters, kernel_size, padding='same', use_bias=False, name='location_conv')
        self.location_layer = tf.layers.Dense( num_units, use_bias=False, dtype=tf.float32, name='location_layer')


    def __call__(self, query, state):
        '''Score the query based on the keys and values.
        This replaces the superclass implementation in order to add in the location term.
        Args:
          query: Tensor of shape `[N, num_units]`.
          state: Tensor of shape `[N, max_inp_len]`
        Returns:
          alignments: Tensor of shape `[N, max_inp_len]`
          next_state: Tensor of shape `[N, max_inp_len]`
        '''
        with tf.variable_scope(None, 'location_sensitive_attention', [query]):
            expanded_alignments = tf.expand_dims(state, axis=2)               # [N, max_inp_len, 1]
            f = self.location_conv(expanded_alignments)                       # [N, max_inp_len, 10]
            processed_location = self.location_layer(f)                       # [N, max_inp_len, num_units]
          
            processed_query = self.query_layer(query) if self.query_layer else query  # [N, num_units]
            processed_query = tf.expand_dims(processed_query, axis=1)         # [N, 1, num_units]
            score = _location_sensitive_score(processed_query, processed_location, self.keys)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state

class DecoderPrenetWrapper(RNNCell):
    '''Runs RNN inputs through a prenet before sending them to the cell.'''
    def __init__(self, cell, is_training, layer_sizes):
        super(DecoderPrenetWrapper, self).__init__()
        self._cell = cell
        self._is_training = is_training
        self._layer_sizes = layer_sizes
  
    @property
    def state_size(self):
        return self._cell.state_size
  
    @property
    def output_size(self):
        return self._cell.output_size
  
    def call(self, inputs, state):
        prenet_out = _create_prenet(inputs, self._is_training, self._layer_sizes, scope='decoder_prenet')
        return self._cell(prenet_out, state)
  
    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

class ConcatOutputAndAttentionWrapper(RNNCell):
    '''Concatenates RNN cell output with the attention context vector.
  
    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    "attention" field that is the context vector.
    '''
    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell
  
    @property
    def state_size(self):
        return self._cell.state_size
  
    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention
  
    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state
  
    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)
  
def _location_sensitive_score(processed_query, processed_location, keys):
    '''Location-sensitive attention score function. 
    Based on _bahdanau_score from tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
    '''
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or array_ops.shape(keys)[2]
    v = tf.get_variable('attention_v', [num_units], dtype=processed_query.dtype)
    return tf.reduce_sum(v * tf.tanh(keys + processed_query + processed_location), [2])


def _create_prenet(inputs, is_training, layer_sizes=[256, 128], scope=None):
    x = inputs
    drop_rate = 0.5 if is_training else 0.0
    with tf.variable_scope(scope or 'prenet'):
        for i, size in enumerate(layer_sizes):
            dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
            x = tf.layers.dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i+1))
    return x

def _create_conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d( inputs,
                                          filters=channels,
                                          kernel_size=kernel_size,
                                          activation=activation,
                                          padding='same')
        return tf.layers.batch_normalization(conv1d_output, training=is_training)

def _create_highwaynet(inputs, scope, depth):
    with tf.variable_scope(scope):
        H = tf.layers.dense( inputs,
                             units=depth,
                             activation=tf.nn.relu,
                             name='H')
        T = tf.layers.dense( inputs,
                             units=depth,
                             activation=tf.nn.sigmoid,
                             name='T',
                             bias_initializer=tf.constant_initializer(-1.0))
        return H * T + inputs * (1.0 - T)

def _create_cbhg(inputs, input_lengths, is_training, scope, K, projections, depth):
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # Convolution bank: concatenate on the last axis to stack channels from all convolutions
            conv_outputs = tf.concat( [_create_conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K+1)],
                                      axis=-1)
  
        # Maxpooling:
        maxpool_output = tf.layers.max_pooling1d(conv_outputs,
                                                 pool_size=2,
                                                 strides=1,
                                                 padding='same')
  
        # Two projection layers:
        proj1_output = _create_conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
        proj2_output = _create_conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')
 
        # Residual connection:
        highway_input = proj2_output + inputs
 
        half_depth = depth // 2
        assert half_depth*2 == depth, 'encoder and postnet depths must be even.'
    
        # Handle dimensionality mismatch:
        if highway_input.shape[2] != half_depth:
            highway_input = tf.layers.dense(highway_input, half_depth)
    
        # 4-layer HighwayNet:
        for i in range(4):
            highway_input = _create_highwaynet(highway_input, 'highway_%d' % (i+1), half_depth)
        rnn_input = highway_input
    
        # Bidirectional RNN
        outputs, states = tf.nn.bidirectional_dynamic_rnn( GRUCell(half_depth),
                                                           GRUCell(half_depth),
                                                           rnn_input,
                                                           sequence_length=input_lengths,
                                                           dtype=tf.float32)
        return tf.concat(outputs, axis=2)  # Concat forward and backward


def _create_encoder_cbhg(inputs, input_lengths, is_training, depth):
    input_channels = inputs.get_shape()[2]
    return _create_cbhg( inputs,
                         input_lengths,
                         is_training,
                         scope='encoder_cbhg',
                         K=16,
                         projections=[128, input_channels],
                         depth=depth)


def _create_post_cbhg(inputs, input_dim, is_training, depth):
    return _create_cbhg( inputs,
                         None,
                         is_training,
                         scope='post_cbhg',
                         K=8,
                         projections=[256, input_dim],
                         depth=depth)
class Tacotron:

    def __init__(self, voice, is_training, eval_batch_size=1):
    
        self.voice      = voice
        self.voice_path = VOICE_PATH % voice
        self.hpfn       = '%s/hparams.json' % self.voice_path
        with codecs.open(self.hpfn, 'r', 'utf8') as hpf:
            self.hp         = json.loads(hpf.read())
        self.batch_size = self.hp['batch_size'] if is_training else eval_batch_size

        max_num_frames  = self.hp['max_iters'] * self.hp['outputs_per_step'] * self.hp['frame_shift_ms'] * self.hp['sample_rate'] / 1000
        n_fft, hop_length, win_length = audio.stft_parameters(self.hp)
        self.max_mfc_frames  = 1 + int((max_num_frames - n_fft) / hop_length)

        # self.inputs        = tf.placeholder(dtype = tf.int32, shape = [None, self.hp['max_inp_len']])
        # self.input_lengths = tf.placeholder(dtype = tf.int32, shape = [None])
        self.inputs        = tf.placeholder(dtype = tf.int32, shape = [self.batch_size, self.hp['max_inp_len']])
        self.input_lengths = tf.placeholder(dtype = tf.int32, shape = [self.batch_size])
        logging.debug('inputs: %s' % self.inputs)
        logging.debug('input_lengths: %s' % self.input_lengths)

        # self.mel_targets    = tf.placeholder(tf.float32, [None, self.max_mfc_frames, self.hp['num_mels']], 'mel_targets')
        # self.linear_targets = tf.placeholder(tf.float32, [None, self.max_mfc_frames, self.hp['num_freq']], 'linear_targets')
        # self.target_lengths = tf.placeholder(tf.int32,   [None],                                           'target_lengths')
        self.mel_targets    = tf.placeholder(tf.float32, [self.batch_size, self.max_mfc_frames, self.hp['num_mels']], 'mel_targets')
        self.linear_targets = tf.placeholder(tf.float32, [self.batch_size, self.max_mfc_frames, self.hp['num_freq']], 'linear_targets')
        self.target_lengths = tf.placeholder(tf.int32,   [self.batch_size],                                           'target_lengths')
        logging.debug('mel_targets: %s' % self.mel_targets)
        logging.debug('linear_targets: %s' % self.linear_targets)
        logging.debug('targets_lengths: %s' % self.target_lengths)

        # Embeddings
        embedding_table = tf.get_variable('embedding', [len(self.hp['alphabet']), self.hp['embed_depth']], dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=0.5))
        logging.debug('embedding_table: %s' % embedding_table)
        embedded_inputs = tf.nn.embedding_lookup(embedding_table, self.inputs)                   # [N, max_inp_len, 256]

        logging.debug('embedded_inputs: %s' % embedded_inputs)

        # Encoder
        prenet_outputs  = _create_prenet(embedded_inputs, is_training, self.hp['prenet_depths']) # [N, max_inp_len, 128]
        logging.debug('prenet_outputs: %s' % prenet_outputs)

        encoder_outputs = _create_encoder_cbhg(prenet_outputs, self.input_lengths, is_training,  # [N, max_inp_len, 256]
                                               self.hp['encoder_depth'])
        logging.debug('encoder_outputs: %s' % encoder_outputs)

        # Attention

        attention_cell = AttentionWrapper( GRUCell(self.hp['attention_depth']),
                                           BahdanauAttention(self.hp['attention_depth'], encoder_outputs),
                                           alignment_history=True,
                                           output_attention=False)                  # [N, T_in, attention_depth=256]
        logging.debug('attention_cell: %s' % attention_cell)
          
        # Apply prenet before concatenation in AttentionWrapper.
        attention_cell = DecoderPrenetWrapper(attention_cell, is_training, self.hp['prenet_depths'])
        logging.debug('attention_cell: %s' % attention_cell)

        # Concatenate attention context vector and RNN cell output into a 512D vector.
        concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)              # [N, max_inp_len, 512]
        logging.debug('concat_cell: %s' % concat_cell)

        # Decoder (layers specified bottom to top):
        decoder_cell = MultiRNNCell([ OutputProjectionWrapper(concat_cell, 256),
                                      ResidualWrapper(GRUCell(256)),
                                      ResidualWrapper(GRUCell(256))
                                    ], state_is_tuple=True)                        # [N, max_inp_len, 256]
        logging.debug('decoder_cell: %s' % decoder_cell)

        # T_in                               -> max_inp_len
        # M           -> hp.num_mels
        # r           -> hp.outputs_per_step
        #                mel_targets         -> frame_targets
        #                max_iters           -> max_iters

        # Project onto r mel spectrograms (predict r outputs at each RNN step):
        output_cell = OutputProjectionWrapper(decoder_cell, self.hp['num_mels'] * self.hp['outputs_per_step'])
        logging.debug('output_cell: %s' % output_cell)

        decoder_init_state = output_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        logging.debug('decoder_init_state: %s' % repr(decoder_init_state))

        if is_training:
            helper = TacoTrainingHelper(self.inputs, self.mel_targets, self.hp['num_mels'], self.hp['outputs_per_step'], self.target_lengths)
        else:
            helper = TacoTestHelper(self.batch_size, self.hp['num_mels'], self.hp['outputs_per_step'])
        logging.debug('helper: %s' % helper)

        (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
          BasicDecoder(output_cell, helper, decoder_init_state),
          maximum_iterations=self.hp['max_iters'])                                 # [N, T_out/r, M*r]
        logging.debug('decoder_outputs: %s' % decoder_outputs)
        logging.debug('final_decoder_state: %s' % repr(final_decoder_state))

        # Reshape outputs to be one output per entry
        self.mel_outputs = tf.reshape(decoder_outputs, [self.batch_size, -1, self.hp['num_mels']])   # [N, T_out, M]
        logging.debug('mel_outputs: %s' % self.mel_outputs)

        # Add post-processing CBHG:
        post_outputs = _create_post_cbhg(self.mel_outputs,                         # [N, T_out, postnet_depth=256]
                                         self.hp['num_mels'], 
                                         is_training,
                                         self.hp['postnet_depth'])
        logging.debug('post_outputs: %s' % post_outputs)
        self.linear_outputs = tf.layers.dense(post_outputs, self.hp['num_freq'])   # [N, T_out, F]
        logging.debug('linear_outputs: %s' % self.linear_outputs)

        # Grab alignments from the final decoder state:
        self.alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])
        logging.debug('alignments: %s' % self.alignments)

        if is_training:

            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.variable_scope('loss') as scope:
                mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
                l1 = tf.abs(self.linear_targets - self.linear_outputs)
                # Prioritize loss for frequencies under 3000 Hz.
                n_priority_freq = int(3000 / (self.hp['sample_rate'] * 0.5) * self.hp['num_freq'])
                linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])
                self.loss = mel_loss + linear_loss

            with tf.variable_scope('optimizer') as scope:
                learning_rate = tf.train.exponential_decay(self.hp['initial_learning_rate'], 
                                                           self.global_step, 
                                                           self.hp['learning_rate_decay_halflife'], 
                                                           0.5)
                optimizer = tf.train.AdamOptimizer(learning_rate, self.hp['adam_beta1'], self.hp['adam_beta2'])
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

        self.sess  = tf.Session()

        self.cpfn  = '%s/model' % self.voice_path

        if os.path.exists('%s.index' % self.cpfn):
            logging.debug ('restoring variables from %s ...' % self.cpfn)
            self.saver.restore(self.sess, self.cpfn)
        else:
            if is_training:
                self.sess.run(tf.global_variables_initializer())
            else:
                raise Exception ("couldn't load model from %s" % self.cpfn)
                

    def _decode_input(self, x):

        res = u''

        for c in x:
            if c:
                res += self.hp['alphabet'][c]

        return res

    def _encode_input (self, txt, idx, input_data, input_lengths):

        ts = cleanup_text (txt, self.hp['lang'], self.hp['alphabet'])

        logging.debug(u'ts=%s' % ts)

        for j, c in enumerate(ts):
            c_enc = self.hp['alphabet'].find(c)
            if c_enc<0:
                logging.error('missing char in alphabet: %s' % c)
                c_enc = self.hp['alphabet'].find(u' ')

            input_data[idx, j] = c_enc

        ts = self._decode_input(input_data[idx])
        logging.debug(u'decoded input=%s' % ts)

        # import pdb; pdb.set_trace()

        input_lengths[idx] = len(ts) + 1 # +1 for start symbol

    def say(self, txt):

        time_start = time()

        logging.debug(u'%fs synthesizing %s' % (time()-time_start, txt))

        input_data     = np.zeros( (1, self.hp['max_inp_len']), dtype='int32')
        input_lengths  = np.zeros( (1, ), dtype='int32')

        logging.debug('input_data.shape=%s, input_lengths.shape=%s' % (input_data.shape, input_lengths.shape))

        self._encode_input(txt, 0, input_data, input_lengths)

        logging.debug('input_data=%s input_lengths=%s' % (input_data[0], input_lengths[0]))

        np.save('say_x', input_data[0])
        logging.debug ('say_x.npy written.')
        np.save('say_xl', input_lengths[0])
        logging.debug ('say_xl.npy written.')

        logging.debug(u'%fs self.session.run...' % (time()-time_start))
        spectrograms = self.sess.run(fetches   = self.linear_outputs,
                                     feed_dict = {
                                                  self.inputs       : input_data,
                                                  self.input_lengths: input_lengths,
                                                 }
                                     )
        spectrogram = spectrograms[0]

        logging.debug('spectrogram.shape=%s' % repr(spectrogram.shape))

        np.save('say_spectrogram', spectrogram)
        logging.debug ('say_spectrogram.npy written.')

        # np.set_printoptions(threshold=np.inf)

        logging.debug(u'%fs audio.inv_spectrogram...' % (time()-time_start))
        wav = audio.inv_spectrogram(spectrogram.T, self.hp)

        logging.debug(u'%fs audio.find_endpoint...' % (time()-time_start))

        logging.debug(u'%fs wav...' % (time()-time_start))
        audio_endpoint = audio.find_endpoint(wav, self.hp)
        # FIXME: wav = wav[:audio_endpoint]

        return wav

    def _load_batch(self, batch_idx):

        self.batch_x       = np.load(DSFN_X  % (self.voice, batch_idx))
        self.batch_xl      = np.load(DSFN_XL % (self.voice, batch_idx))
        self.batch_ys      = np.load(DSFN_YS % (self.voice, batch_idx))
        self.batch_ym      = np.load(DSFN_YM % (self.voice, batch_idx))
        self.batch_yl      = np.load(DSFN_YL % (self.voice, batch_idx))

        ts = self._decode_input(self.batch_x[0])
        logging.debug(u'ts %d %s' % (batch_idx, ts))

    def _plot_alignment(self, alignment, path, info=None):
        fig, ax = plt.subplots()
        im = ax.imshow( alignment,
                        aspect='auto',
                        origin='lower',
                        interpolation='none')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        if info is not None:
            xlabel += '\n\n' + info
        plt.xlabel(xlabel)
        plt.ylabel('Encoder timestep')
        plt.tight_layout()
        plt.savefig(path, format='png')

    def train(self, num_epochs=DEFAULT_NUM_EPOCHS):

        logging.info ('counting numpy batches...')

        num_batches = 0
        while True:
            if os.path.exists(DSFN_X % (self.voice, num_batches)):
                num_batches += 1
            else:
                break

        logging.info ('counting numpy batches... %d batches found.' % num_batches)

        if DEBUG_LIMIT:
            logging.warn ('limiting number of batches to %d for debugging' % DEBUG_LIMIT)
            num_batches = DEBUG_LIMIT

        self._load_batch(0) # make sure we have one batch loaded at all times so batch_*.shape works

        batch_size = self.hp['batch_size']

        batch_idxs = range(0, num_batches)

        for epoch in range(num_epochs):

            random.shuffle(batch_idxs)
            epoch_loss = 0

            for i, batch_idx in enumerate(batch_idxs):

                self._load_batch(batch_idx)

                step_out, loss_out, opt_out, spectrogram, alignment = self.sess.run([self.global_step, self.loss, self.optimize, self.linear_outputs, self.alignments],
                                                                                    feed_dict={self.inputs         : self.batch_x, 
                                                                                               self.input_lengths  : self.batch_xl,
                                                                                               self.mel_targets    : self.batch_ym,
                                                                                               self.linear_targets : self.batch_ys,
                                                                                               self.target_lengths : self.batch_yl})

                epoch_loss += loss_out

                logging.info ('epoch: %5d, step %4d/%4d batch #%4d, loss: %7.5f, avg loss: %7.5f' % (epoch, i+1, num_batches, batch_idx, loss_out, epoch_loss / (i+1)))

            cpfn = CHECKPOINT_FN % (self.voice, epoch)
            logging.info('Saving checkpoint to: %s' % cpfn)
            self.saver.save(self.sess, cpfn, global_step=step_out)

            logging.info('Saving audio and alignment...')

            # import pdb; pdb.set_trace()

            # input_seq, spectrogram, alignment = sess.run([inputs, input_lengths, linear_outputs, alignments],
            #                                              feed_dict={inputs         : eval_x,
            #                                                         input_lengths  : eval_xl,
            #                                                         mel_targets    : eval_ym,
            #                                                         linear_targets : eval_ys})

            waveform = audio.inv_spectrogram(spectrogram[0].T, self.hp)

            wavfn = WAV_FN % (self.voice, epoch)
            audio.save_wav(waveform, wavfn, self.hp)
            logging.info('%s written.' % wavfn)

            specfn = SPEC_FN % (self.voice, epoch)
            cmd = 'sox %s -n spectrogram -o %s' % (wavfn, specfn)
            logging.info(cmd)
            os.system(cmd)

            # import pdb; pdb.set_trace()

            plotfn = ALIGN_FN % (self.voice, epoch)
            self._plot_alignment(alignment[0], plotfn,
                                 info='epoch=%d, loss=%.5f' % (epoch, loss_out))
            logging.info ('alignment %s plotted to %s' % (alignment[0].shape, plotfn) )



    def eval_batch(self, batch_idx):

        self._load_batch(batch_idx)

        time_start = time()

        logging.debug('input_data.shape=%s, input_lengths.shape=%s' % (self.batch_x.shape, self.batch_xl.shape))

        logging.debug('x[0]=%s xl[0]=%s' % (self.batch_x[0], self.batch_xl[0]))

        np.save('eval_x', self.batch_x[0])
        logging.debug ('eval_x.npy written.')
        np.save('eval_xl', self.batch_xl[0])
        logging.debug ('eval_xl.npy written.')

        logging.debug(u'%fs self.session.run...' % (time()-time_start))
        spectrograms = self.sess.run(fetches   = self.linear_outputs,
                                     feed_dict = {
                                                  self.inputs       : self.batch_x,
                                                  self.input_lengths: self.batch_xl,
                                                 }
                                     )
        spectrogram = spectrograms[0]

        logging.debug('spectrogram.shape=%s' % repr(spectrogram.shape))
        logging.debug('batch_ys.shape=%s' % repr(self.batch_ys.shape))

        np.save('eval_spectrogram', spectrogram)
        logging.debug ('eval_spectrogram.npy written.')

        # np.set_printoptions(threshold=np.inf)

        logging.debug(u'%fs audio.inv_spectrogram...' % (time()-time_start))
        wav = audio.inv_spectrogram(spectrogram.T, self.hp)

        logging.debug(u'%fs audio.find_endpoint...' % (time()-time_start))

        logging.debug(u'%fs wav...' % (time()-time_start))
        audio_endpoint = audio.find_endpoint(wav, self.hp)
        # FIXME: wav = wav[:audio_endpoint]

        return wav

