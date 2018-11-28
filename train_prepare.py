#!/usr/bin/env python
# -*- coding: utf-8 -*- 

#
# Copyright 2017, 2018 Guenter Bartsch
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

#
# prepare tacotron datasets
#


import sys
import re
import os
import StringIO
import ConfigParser
import struct
import wave
import codecs
import logging
import random
import json

import numpy      as np

from optparse          import OptionParser
from nltools           import misc
from zamiatts          import WAV_PATH, DSFN_PATH, HPARAMS_SRC, DSFN_X, DSFN_XL, DSFN_YS, DSFN_YM, DSFN_YL, VOICE_PATH, cleanup_text
from zamiatts          import audio

DEBUG_LIMIT  = 0
# DEBUG_LIMIT = 65
# DEBUG_LIMIT = 512

PROC_TITLE      = 'train_prepare'

def _decode_input(x):

    global hparams

    res = u''

    for c in x:
        if c:
            res += hparams['alphabet'][c]

    return res

#
# init terminal
#

misc.init_app (PROC_TITLE)

#
# command line
#

parser = OptionParser("usage: %prog [options] <voice_in> <voice_out>")

parser.add_option ("-g", "--gender", dest="gender", type = "str", default="male",
                   help="gender (default: male)")

parser.add_option ("-l", "--lang", dest="lang", type = "str", default="de_DE",
                   help="language (default: de_DE)")

parser.add_option("-v", "--verbose", action="store_true", dest="verbose", 
                  help="enable debug output")


(options, args) = parser.parse_args()

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

if len(args) != 2:
    parser.print_help()
    sys.exit(0)

voice_in   = args[0]
voice_out  = args[1]
lang       = options.lang.split('_')[0]
mailabsdir = '/home/bofh/projects/ai/data/speech/corpora/m_ailabs/%s/by_book/%s/%s' % (options.lang, options.gender, voice_in)

#
# clean up / setup directories
#

cmd = 'rm -rf %s' % (WAV_PATH % voice_out)
logging.info(cmd)
os.system(cmd)

cmd = 'mkdir -p %s' % (WAV_PATH % voice_out)
logging.info(cmd)
os.system(cmd)

cmd = 'rm -rf %s' % (DSFN_PATH % voice_out)
logging.info(cmd)
os.system(cmd)

cmd = 'mkdir -p %s' % (DSFN_PATH % voice_out)
logging.info(cmd)
os.system(cmd)

#
# globals
#

with codecs.open(HPARAMS_SRC, 'r', 'utf8') as hpf:
    hparams         = json.loads(hpf.read())
max_inp_len     = hparams['max_inp_len']
max_num_frames  = hparams['max_iters'] * hparams['outputs_per_step'] * hparams['frame_shift_ms'] * hparams['sample_rate'] / 1000

n_fft, hop_length, win_length = audio.stft_parameters(hparams)
max_mfc_frames  = 1 + int((max_num_frames - n_fft) / hop_length)

logging.info ('max_mfc_frames=%d, num_freq=%d, num_mels=%d' % (max_mfc_frames,hparams['num_freq'],hparams['num_mels']))

#
# read wav files
#

training_data  = []
num_skipped = 0

for book in os.listdir(mailabsdir):
    logging.info('extracting training data from book %s' % book)

    metafn = '%s/%s/metadata_mls.json' % (mailabsdir, book)
    if not os.path.exists(metafn):
        continue

    with codecs.open(metafn, 'r', 'utf8') as metaf:
        meta = json.loads(metaf.read())

    # print repr(meta)

    for wavfn in meta:

        ts_orig = meta[wavfn]['clean']

        ts = cleanup_text(ts_orig, lang, hparams['alphabet'])

        logging.debug(u'ts_orig %s' % ts_orig)
        logging.debug(u'ts      %s' % ts)

        if len(ts) > (max_inp_len-1):
            num_skipped += 1
            pskipped = num_skipped * 100 / (len(training_data) + num_skipped)
            logging.error('%6d %-20s: transcript too long (%4d > %4d) %3d%% skipped' % (len(training_data), wavfn, len(ts), max_inp_len, pskipped))
            continue

        wav_path = '%s/%s/wavs/%s' % (mailabsdir, book, wavfn)

        tmp_wav = (WAV_PATH % voice_out) + '/' + wavfn

        # cmd = 'sox %s %s silence -l 1 0.1 1%% -1 2.0 1%% compand 0.02,0.20 5:-60,-40,-10 -5 -90 0.1' % (wav_path, tmp_wav)
        cmd = 'sox %s -r 16000 -b 16 -c 1 %s silence 1 0.15 0.5%% reverse silence 1 0.15 0.5%% reverse gain -n -3' % (wav_path, tmp_wav)
        logging.debug(cmd)
        os.system(cmd)

        wav = audio.load_wav(tmp_wav)
        # FIXME: remove, we're using sox for this now wav = audio.trim_silence(wav, hparams)

        # logging.debug('wav: %s' % wav.shape)
        if wav.shape[0] < 512:
            num_skipped += 1
            pskipped = num_skipped * 100 / (len(training_data) + num_skipped)
            logging.error('%6d %-20s: audio too short (%4d < 512) %3d%% skipped' % (len(training_data), wavfn, len(ts), pskipped))
            continue
 
        spectrogram     = audio.spectrogram(wav, hparams).astype(np.float32)
        mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)

        if spectrogram.shape[1] > (max_mfc_frames-1):
            num_skipped += 1
            pskipped = num_skipped * 100 / (len(training_data) + num_skipped)
            logging.error('%6d %-20s: audio too long (%4d > %4d) %3d%% skipped' % (len(training_data), wavfn, spectrogram.shape[1], max_mfc_frames, pskipped))
            continue

        logging.info('%6d %-20s: ok, spectrogram.shape=%s, mel_spectrogram.shape=%s' % (len(training_data), wavfn, spectrogram.shape, mel_spectrogram.shape))
        training_data.append((ts, spectrogram.T, mel_spectrogram.T))

        if DEBUG_LIMIT and len(training_data) >= DEBUG_LIMIT:
            break

    if DEBUG_LIMIT and len(training_data) >= DEBUG_LIMIT:
        break

random.shuffle(training_data)

logging.info ('training data: %d samples (%d skipped), max_inp_len=%d, max_frames_len=%d' % (len(training_data), num_skipped, max_inp_len, max_num_frames))

#
# create numpy datasets
#

logging.info ('generating numpy arrays. max_mfc_frames=%d' % max_mfc_frames)

batch_size = hparams['batch_size']

input_data     = np.zeros( (batch_size, max_inp_len), dtype='int32')
input_lengths  = np.zeros( (batch_size, ), dtype='int32')
target_data_s  = np.zeros( (batch_size, max_mfc_frames, hparams['num_freq']) , dtype='float32')
target_data_m  = np.zeros( (batch_size, max_mfc_frames, hparams['num_mels']) , dtype='float32')
target_lengths = np.zeros( (batch_size, ), dtype='int32')

for i, (ts, S, M) in enumerate(training_data):

    batch_idx = i % batch_size
    batch_num = i / batch_size

    input_data[batch_idx].fill(0)

    # logging.debug(u'transcript: %s' % ts)

    for j, c in enumerate(ts):
        c_enc = hparams['alphabet'].find(c)
        if c_enc<0:
            logging.error('missing char in alphabet: %s' % c)
            # c_enc = hparams['alphabet'].find(u' ')

        input_data[batch_idx, j] = c_enc


    ts = _decode_input(input_data[batch_idx])

    input_lengths[batch_idx] = len(ts) + 1 # +1 for start symbol

    target_data_s[batch_idx]  = np.pad(S, ((0, max_mfc_frames - S.shape[0]), (0,0)), 'constant', constant_values=(0.0,0.0))
    target_data_m[batch_idx]  = np.pad(M, ((0, max_mfc_frames - S.shape[0]), (0,0)), 'constant', constant_values=(0.0,0.0))
    target_lengths[batch_idx] = S.shape[0] + 1

    logging.debug(u'batch_idx=%4d, batch_num=%4d %s' % (batch_idx, batch_num, ts[:64]))

    if batch_idx == (batch_size-1):

        np.save(DSFN_X % (voice_out, batch_num), input_data)
        logging.info("%s written. %s" % (DSFN_X % (voice_out, batch_num), input_data.shape))

        np.save(DSFN_XL % (voice_out, batch_num), input_lengths)
        logging.info("%s written. %s" % (DSFN_XL % (voice_out, batch_num), input_lengths.shape))

        np.save(DSFN_YS % (voice_out, batch_num), target_data_s)
        logging.info("%s written. %s" % (DSFN_YS % (voice_out, batch_num), target_data_s.shape))

        np.save(DSFN_YM % (voice_out, batch_num), target_data_m)
        logging.info("%s written. %s" % (DSFN_YM % (voice_out, batch_num), target_data_m.shape))

        np.save(DSFN_YL % (voice_out, batch_num), target_lengths)
        logging.info("%s written. %s" % (DSFN_YL % (voice_out, batch_num), target_lengths.shape))

