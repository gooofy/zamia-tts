#!/usr/bin/env python
# -*- coding: utf-8 -*- 

#
# Copyright 2018 Guenter Bartsch
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

import os
import re
import sys
import logging
import json
import codecs
import readline
import tempfile

import scipy.io.wavfile
import numpy as np

from optparse             import OptionParser

from nltools              import misc, pulseplayer
from zamiatts.tacotron    import Tacotron
from zamiatts             import audio

PROC_TITLE      = 'eval'

NPDIR = '../speech/data/dst/tts/training/linda/'
WAV_FN          = 'eval.wav'
VOICE_PATH      = 'data/model/%s'
# VOICE           = 'voice-karlsson-latest'
DEFAULT_VOICE   = 'voice-linda-latest'
BATCH_SIZE      = 32

# x_13036.npy (1, 240) these figures suggest that the agents of the secret service are substantially overworked .

X_FN  = NPDIR + 'x_13036.npy'
X_TXT = 'These figures suggest that the agents of the secret service are substantially overworked.'
X_L   = len(X_TXT)

#
# init
#

misc.init_app(PROC_TITLE)

#
# command line
#

parser = OptionParser("usage: %prog [options] [<text>]")

parser.add_option("-V", "--voice", dest="voice", type = "str", default=DEFAULT_VOICE,
                  help="voice, default: %s" % DEFAULT_VOICE)

parser.add_option("-v", "--verbose", action="store_true", dest="verbose", 
                  help="enable debug output")

(options, args) = parser.parse_args()

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

#
# main
#

taco = Tacotron(options.voice, is_training=False, eval_batch_size=32, voice_path=VOICE_PATH, write_debug_files=True)

if taco.hp['batch_size'] != BATCH_SIZE:
    logging.error('incorrect batch size, is %d should be %d.' % (BATCH_SIZE, taco.hp['batch_size']))
    sys.exit(1)

# for fn in os.listdir(NPDIR):
# 
#     if not fn.startswith ('x_'):
#         continue
# 
#     x  = np.load(NPDIR+'/'+fn)
# 
#     txt = taco.decode_input(x[0])
# 
#     print fn, x.shape, txt

x = np.load(X_FN)[0]

print x

print repr(taco.hp)

batch_x  = np.zeros( (BATCH_SIZE, taco.hp['max_inp_len']), dtype='int32')
batch_xl = np.zeros( (BATCH_SIZE, ), dtype='int32')

for i in range(BATCH_SIZE):
    batch_x[i] = x
    batch_xl[i] = X_L

print batch_x
print batch_xl

wav = taco.eval_batch(batch_x, batch_xl)

audio.save_wav(wav, WAV_FN, taco.hp)
logging.info("%s written." % WAV_FN)

