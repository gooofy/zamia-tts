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
from zamiatts          import DSFN_X, DSFN_XL, DSFN_YS, DSFN_YM, DSFN_YL, cleanup_text
from zamiatts          import audio

DEBUG_LIMIT  = 0
# DEBUG_LIMIT = 65

PROC_TITLE      = 'train_prepare'

VOICE           = 'karlsson'
GENDER          = 'male'
MAILABSDIR      = '/home/bofh/projects/ai/data/speech/corpora/m_ailabs_de/de_DE/by_book/%s/%s' % (GENDER, VOICE)
LANG            = 'de'

TMP_WAV         = 'tmp/tmp.wav'

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

parser = OptionParser("usage: %prog [options])")

parser.add_option("-v", "--verbose", action="store_true", dest="verbose", 
                  help="enable debug output")


(options, args) = parser.parse_args()

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

#
# globals
#

with codecs.open('voices/%s/hparams.json' % VOICE, 'r', 'utf8') as hpf:
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

for book in os.listdir(MAILABSDIR):
    logging.info('extracting training data from book %s' % book)

    metafn = '%s/%s/metadata_mls.json' % (MAILABSDIR, book)
    if not os.path.exists(metafn):
        continue

    with codecs.open(metafn, 'r', 'utf8') as metaf:
        meta = json.loads(metaf.read())

    # print repr(meta)

    for wavfn in meta:

        ts_orig = meta[wavfn]['clean']

        ts = cleanup_text(ts_orig, LANG, hparams['alphabet'])

        logging.debug(u'ts_orig %s' % ts_orig)
        logging.debug(u'ts      %s' % ts)

        if len(ts) > (max_inp_len-1):
            num_skipped += 1
            pskipped = num_skipped * 100 / (len(training_data) + num_skipped)
            logging.error('%6d %-20s: transcript too long (%4d > %4d) %3d%% skipped' % (len(training_data), wavfn, len(ts), max_inp_len, pskipped))
            continue

        wav_path = '%s/%s/wavs/%s' % (MAILABSDIR, book, wavfn)

        cmd = 'sox %s %s compand 0.02,0.20 5:-60,-40,-10 -5 -90 0.1' % (wav_path, TMP_WAV)
        logging.debug(cmd)
        os.system(cmd)

        wav = audio.load_wav(TMP_WAV)
        wav = audio.trim_silence(wav, hparams)

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

        np.save(DSFN_X % (VOICE, batch_num), input_data)
        logging.info("%s written. %s" % (DSFN_X % (VOICE, batch_num), input_data.shape))

        np.save(DSFN_XL % (VOICE, batch_num), input_lengths)
        logging.info("%s written. %s" % (DSFN_XL % (VOICE, batch_num), input_lengths.shape))

        np.save(DSFN_YS % (VOICE, batch_num), target_data_s)
        logging.info("%s written. %s" % (DSFN_YS % (VOICE, batch_num), target_data_s.shape))

        np.save(DSFN_YM % (VOICE, batch_num), target_data_m)
        logging.info("%s written. %s" % (DSFN_YM % (VOICE, batch_num), target_data_m.shape))

        np.save(DSFN_YL % (VOICE, batch_num), target_lengths)
        logging.info("%s written. %s" % (DSFN_YL % (VOICE, batch_num), target_lengths.shape))

