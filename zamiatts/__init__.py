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

from nltools.tokenizer import tokenize

WAV_PATH        = 'data/dst/tts/wav/%s'

DSFN_PATH       = 'data/dst/tts/training/%s'
DSFN_X          = 'data/dst/tts/training/%s/x_%04d.npy'
DSFN_XL         = 'data/dst/tts/training/%s/xl_%04d.npy'
DSFN_YS         = 'data/dst/tts/training/%s/ys_%04d.npy'
DSFN_YM         = 'data/dst/tts/training/%s/ym_%04d.npy'
DSFN_YL         = 'data/dst/tts/training/%s/yl_%04d.npy'
HPARAMS_SRC     = 'data/src/tts/hparams.json'

VOICE_PATH      = 'data/dst/tts/voices/%s'
HPARAMS_FN      = 'data/dst/tts/voices/%s/hparams.json'
CHECKPOINT_DIR  = 'data/dst/tts/voices/%s/cp'
CHECKPOINT_FN   = 'data/dst/tts/voices/%s/cp/cp%04d'
EVAL_DIR        = 'data/dst/tts/voices/%s/eval'
WAV_FN          = 'data/dst/tts/voices/%s/eval/wav_%04d.wav'
SPEC_FN         = 'data/dst/tts/voices/%s/eval/spec_%04d.png'
ALIGN_FN        = 'data/dst/tts/voices/%s/eval/align_%04d.png'

def cleanup_text (txt, lang, alphabet):

    tokens = map (lambda w: filter (lambda a: a in alphabet, w), tokenize(txt, lang=lang, keep_punctuation=True))

    return u' '.join(filter (lambda x: len(x)>0, tokens))

