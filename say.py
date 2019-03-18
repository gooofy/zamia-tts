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
# synthesize speech using a tacotron model, output through pulseaudio
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

PROC_TITLE      = 'say'

VOICE_PATH      = 'data/model/%s'
# VOICE           = 'voice-karlsson-latest'
DEFAULT_VOICE   = 'voice-linda-latest'

def synthesize(txt):

    global taco, options, player

    logging.info('Synthesizing: %s' % txt)
    wav = taco.say(txt, trim_silence=(not options.untrimmed_output), dyn_range_compress=options.dyn_range_compress)

    if options.wavfn:
        audio.save_wav(wav, options.wavfn, taco.hp)
        logging.info("%s written." % options.wavfn)

    else:

        with tempfile.NamedTemporaryFile() as temp:
            audio.save_wav(wav, temp.name, taco.hp)
            temp.flush()
            temp.seek(0)

            wav = temp.read()

            logging.debug("%s written (%d bytes)." % (temp.name, len(wav)))

            player.play(wav, async=True)

#
# init
#

misc.init_app(PROC_TITLE)
readline.set_history_length(1000)

#
# command line
#

parser = OptionParser("usage: %prog [options] [<text>]")

parser.add_option("-o", "--output-wav", dest="wavfn", type = "str", 
                  help="output wav filename, default: output through pulseaudio")

parser.add_option("-V", "--voice", dest="voice", type = "str", default=DEFAULT_VOICE,
                  help="voice, default: %s" % DEFAULT_VOICE)

parser.add_option("-d", "--dyn-range", action="store_true", dest="dyn_range_compress", 
                  help="enable dynamic range compression")

parser.add_option("-u", "--untrimmed-output", action="store_true", dest="untrimmed_output", 
                  help="disable silence trimming")

parser.add_option("-v", "--verbose", action="store_true", dest="verbose", 
                  help="enable debug output")

(options, args) = parser.parse_args()

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

if not options.wavfn:
    player = pulseplayer.PulsePlayer(PROC_TITLE)

#
# main
#

taco = Tacotron(options.voice, is_training=False, voice_path=VOICE_PATH, tf_device='/cpu:0')

if len(args)>0:

    for i, raw_txt in enumerate(args):
        txt = raw_txt.decode('utf8')
        synthesize(txt)

else:

    while True:

        txt = raw_input("%s >" % options.voice)

        txt = txt.strip()
        if not txt:
            break

        synthesize(txt.decode('utf8'))
       
 
