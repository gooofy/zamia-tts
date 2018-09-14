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

# import librosa
# import librosa.filters
import math
import six
import scipy
import scipy.signal
import numpy as np
import tensorflow as tf
import scipy.fftpack as fft
from hparams import hparams
from numpy.lib.stride_tricks import as_strided

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2**8 * 2**10

def spectrogram(y):
  D = _stft(y)
  S = _amp_to_db(np.abs(D)) - hparams.ref_level_db
  return _normalize(S)


def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) +
                 hparams.ref_level_db)  # Convert back to linear
  # Reconstruct phase
  return _griffin_lim(S ** hparams.power)


# def inv_spectrogram_tensorflow(spectrogram):
#   '''Builds computational graph to convert spectrogram to waveform using TensorFlow.'''
#   import pdb; pdb.set_trace()
#   S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hparams.ref_level_db)
#   return _griffin_lim_tensorflow(tf.pow(S, hparams.power))


def melspectrogram(y):
  D = _stft(y)
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
  return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.5):
  window_length = int(hparams.sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    ww = wav[x:x + window_length]
    # for i, sample in enumerate(ww):
    #     print '%6.3f' % sample,
    #     if i % 12 == 11:
    #         print
    # print
    # print "---->", x, x + window_length, ww.shape, np.max(ww), threshold
    # print
    if np.max(ww) < threshold:
      return x + hop_length
  return len(wav)


def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(hparams.griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y


# def _griffin_lim_tensorflow(S):
#   '''TensorFlow implementation of Griffin-Lim
#   Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
#   '''
#   with tf.variable_scope('griffinlim'):
#     # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
#     S = tf.expand_dims(S, 0)
#     S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
#     y = _istft_tensorflow(S_complex)
#     for i in range(hparams.griffin_lim_iters):
#       est = _stft_tensorflow(y)
#       angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
#       y = _istft_tensorflow(S_complex * angles)
#     return tf.squeeze(y, 0)

def pad_center(data, size, axis=-1, **kwargs):

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be '
                              'at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)

def valid_audio(y, mono=True):

    if not isinstance(y, np.ndarray):
        raise ParameterError('data must be of type numpy.ndarray')

    if not np.issubdtype(y.dtype, np.floating):
        raise ParameterError('data must be floating-point')

    if mono and y.ndim != 1:
        raise ParameterError('Invalid shape for monophonic audio: '
                             'ndim={:d}, shape={}'.format(y.ndim, y.shape))

    elif y.ndim > 2 or y.ndim == 0:
        raise ParameterError('Audio must have shape (samples,) or (channels, samples). '
                             'Received shape={}'.format(y.shape))

    if not np.isfinite(y).all():
        raise ParameterError('Audio buffer is not finite everywhere')

    return True

def frame(y, frame_length=2048, hop_length=512):
    if not isinstance(y, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(y)={}'.format(type(y)))

    if y.ndim != 1:
        raise ParameterError('Input must be one-dimensional, '
                             'given y.ndim={}'.format(y.ndim))

    if len(y) < frame_length:
        raise ParameterError('Buffer is too short (n={:d})'
                             ' for frame_length={:d}'.format(len(y), frame_length))

    if hop_length < 1:
        raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))

    if not y.flags['C_CONTIGUOUS']:
        raise ParameterError('Input buffer must be contiguous.')

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames

def tiny(x):

    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny

def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                     stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]]

    return stft_matrix

def get_window(window, Nx, fftbins=True):
    if six.callable(window):
        return window(Nx)

    elif (isinstance(window, (six.string_types, tuple)) or
          np.isscalar(window)):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ParameterError('Window size mismatch: '
                             '{:d} != {:d}'.format(len(window), Nx))
    else:
        raise ParameterError('Invalid window specification: {}'.format(window))

def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]]

    return stft_matrix

def fix_length(data, size, axis=-1, **kwargs):

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data

def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):

    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ParameterError('threshold={} must be strictly '
                             'positive'.format(threshold))

    if fill not in [None, False, True]:
        raise ParameterError('fill={} must be None or boolean'.format(fill))

    if not np.all(np.isfinite(S)):
        raise ParameterError('Input must be finite')

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError('Cannot normalize with norm=0 and fill=True')

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True)**(1./norm)

        if axis is None:
            fill_norm = mag.size**(-1./norm)
        else:
            fill_norm = mag.shape[axis]**(-1./norm)

    elif norm is None:
        return S

    else:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm

def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
    '''Helper function for window sum-square calculation.'''

    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]

def window_sumsquare(window, n_frames, hop_length=512, win_length=None, n_fft=2048,
                     dtype=np.float32, norm=None):
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = normalize(win_sq, norm=norm)**2
    win_sq = pad_center(win_sq, n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x

def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, dtype=np.float32, length=None):
    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft
    ifft_window = pad_center(ifft_window, n_fft)

    n_frames = stft_matrix.shape[1]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)

    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, i].flatten()
        spec = np.concatenate((spec, spec[-2:0:-1].conj()), 0)
        ytmp = ifft_window * fft.ifft(spec).real

        y[sample:(sample + n_fft)] = y[sample:(sample + n_fft)] + ytmp

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(window,
                                       n_frames,
                                       win_length=win_length,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       dtype=dtype)

    approx_nonzero_indices = ifft_window_sum > tiny(ifft_window_sum)
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[int(n_fft // 2):-int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = fix_length(y[start:], length)

    return y


def _istft(y):
  _, hop_length, win_length = _stft_parameters()
  return istft(y, hop_length=hop_length, win_length=win_length)

# 
# def _stft_tensorflow(signals):
#   n_fft, hop_length, win_length = _stft_parameters()
#   return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)
# 
# 
# def _istft_tensorflow(stfts):
#   n_fft, hop_length, win_length = _stft_parameters()
#   print stfts, win_length, hop_length, n_fft
#   return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters():
  n_fft = (hparams.num_freq - 1) * 2
  hop_length = int(hparams.frame_shift_ms / 1000.0 * hparams.sample_rate)
  win_length = int(hparams.frame_length_ms / 1000.0 * hparams.sample_rate)
  return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None


def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
  n_fft = (hparams.num_freq - 1) * 2
  return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels,
                             fmin=hparams.min_mel_freq, fmax=hparams.max_mel_freq)


def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
  return np.power(10.0, x * 0.05)


def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _normalize(S):
  return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
  return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


# def _denormalize_tensorflow(S):
#   return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

