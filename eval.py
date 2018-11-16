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

#
# synthesize speech using a tacotron model
#

import os
import re
import sys
import logging
import json
import codecs

import scipy.io.wavfile
import numpy as np

from optparse             import OptionParser

from nltools              import misc
from zamiatts.tacotron    import Tacotron
from zamiatts             import audio

PROC_TITLE      = 'eval'

VOICE           = 'karlsson'

BATCH_IDX       = 384
BATCH_SIZE      = 32

#
# init
#

misc.init_app(PROC_TITLE)

#
# command line
#

parser = OptionParser("usage: %prog [options]")

parser.add_option("-v", "--verbose", action="store_true", dest="verbose", 
                  help="enable debug output")


(options, args) = parser.parse_args()

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

taco = Tacotron(VOICE, is_training=False, eval_batch_size=BATCH_SIZE)

wav = taco.eval_batch(BATCH_IDX)

wavfn = 'eval.wav'
audio.save_wav(wav, wavfn, taco.hp)

logging.info("%s written." % wavfn)

