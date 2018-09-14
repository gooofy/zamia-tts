#!/usr/bin/env python
# -*- coding: utf-8 -*- 

#
# Copyright 2018 Guenter Bartsch
# Copyright 2018 Keith Ito
# Copyright 2018 MycroftAI
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

import io
import math
import logging

from time import time

import numpy      as np
import tensorflow as tf

from zamiatts.hparams import hparams, hparams_debug_string
from zamiatts         import text_to_sequence
from .                import audio
from .tacotron        import Tacotron

class Synthesizer:

    def load(self, checkpoint_path, model_name='tacotron'):
        inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
        with tf.variable_scope('model') as scope:
            self.model = Tacotron(hparams)
            self.model.initialize(inputs, input_lengths)
            self.alignment = self.model.alignments[0]

        logging.info('Loading checkpoint: %s' % checkpoint_path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, text):

        time_start = time()

        logging.debug(u'%fs synthesizing %s' % (time()-time_start, text))

        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        seq = text_to_sequence(text, cleaner_names)

        feed_dict = {
            self.model.inputs: [np.asarray(seq, dtype=np.int32)],
            self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
        }

        logging.debug(u'%fs self.session.run...' % (time()-time_start))
        spectrogram = self.session.run(
            self.model.linear_outputs[0],
            feed_dict=feed_dict
        )

        np.set_printoptions(threshold=np.inf)

        logging.debug(u'%fs audio.inv_spectrogram...' % (time()-time_start))
        wav = audio.inv_spectrogram(spectrogram.T)

        logging.debug(u'%fs audio.find_endpoint...' % (time()-time_start))

        logging.debug(u'%fs wav...' % (time()-time_start))
        audio_endpoint = audio.find_endpoint(wav)
        wav = wav[:audio_endpoint]

        return wav

