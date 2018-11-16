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

PROC_TITLE      = 'say'

VOICE           = 'karlsson'

#
# init
#

misc.init_app(PROC_TITLE)

#
# command line
#

parser = OptionParser("usage: %prog [options] <text>")

parser.add_option("-u", "--untrimmed-output", action="store_true", dest="untrimmed_output", 
                  help="disable silence trimming")

parser.add_option("-v", "--verbose", action="store_true", dest="verbose", 
                  help="enable debug output")


(options, args) = parser.parse_args()

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

if len(args) != 1:
    parser.print_usage()
    sys.exit(1)

# with codecs.open('hparams.json', 'w', 'utf8') as hpf:
#     hpf.write(json.dumps(hparams))


taco = Tacotron(VOICE, is_training=False)

for i, txt in enumerate(args):

    logging.info('Synthesizing: %s' % txt)
    wav = taco.say(txt, trim_silence=(not options.untrimmed_output))

    wavfn = '%d.wav' % i
    audio.save_wav(wav, wavfn, taco.hp)

    # wav16 = (32767*wav).astype(np.int16)
    # scipy.io.wavfile.write(wavfn, taco.hp['sample_rate'], wav16)

    logging.info("%s written." % wavfn)

