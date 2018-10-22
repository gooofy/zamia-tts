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


hparams = {
    # Audio:
    'num_mels'        :    80,
    'num_freq'        :  1025,
    'min_mel_freq'    :   125,
    'max_mel_freq'    :  7600,
    'sample_rate'     : 16000,
    'frame_length_ms' :    50,
    'frame_shift_ms'  :  12.5,
    'min_level_db'    :  -100,
    'ref_level_db'    :    20,
    # Eval:
    # max_iters=200,
    # griffin_lim_iters=50,
    'power'           :   1.5,            # Power to raise magnitudes to prior to Griffin-Lim

    # MAILABS silence trim params
    'trim_fft_size'   :  1024,
    'trim_hop_size'   :   256,
    'trim_top_db'     :    40,

    # # Model:
    # # TODO: add more configurable hparams
    # outputs_per_step=5,
    # embedding_dim=512,

    # # Training:
    # batch_size=32,
    # adam_beta1=0.9,
    # adam_beta2=0.999,
    # initial_learning_rate=0.0015,
    # learning_rate_decay_halflife=100000,

}

