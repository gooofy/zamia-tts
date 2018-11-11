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
# some of the audio processing code originates from librosa
#

import math
import six
import scipy
import scipy.signal
import scipy.io.wavfile
import numpy as np
import scipy.fftpack as fft
import logging
import wave
import struct
from numpy.lib.stride_tricks import as_strided

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2**8 * 2**10

def stft_parameters(hparams):
    n_fft      = (hparams['num_freq'] - 1) * 2
    hop_length = int(hparams['frame_shift_ms']  / 1000.0 * hparams['sample_rate'])
    win_length = int(hparams['frame_length_ms'] / 1000.0 * hparams['sample_rate'])
    return n_fft, hop_length, win_length

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
    _valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = _frame(y, frame_length=n_fft, hop_length=hop_length)

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

def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, dtype=np.float32):
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
    # print 'istft: expected_signal_len:', expected_signal_len
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

    if center:
        y = y[int(n_fft // 2):-int(n_fft // 2)]

    return y

def _griffin_lim(S, hparams):
    n_fft, hop_length, win_length = stft_parameters(hparams)
    griffin_lim_iters = hparams['griffin_lim_iters']

    # print '_griffin_lim: S.shape:', S.shape

    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    # print '_griffin_lim: angles.shape:', angles.shape

    S_complex = np.abs(S).astype(np.complex)
    y = istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
    # print '_griffin_lim: y.shape:', y.shape
    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)))
        # print '_griffin_lim: angles.shape:', angles.shape
        y = istft(S_complex * angles, hop_length=hop_length, win_length=win_length)

    return y

def inv_spectrogram(spectrogram, hparams):
    S = _db_to_amp(_denormalize(spectrogram, hparams) + hparams['ref_level_db'])  # Convert back to linear
    # Reconstruct phase
    return _griffin_lim(S ** hparams['power'], hparams)

def spectrogram(y, hparams):
    n_fft, hop_length, win_length = stft_parameters(hparams)
    D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S = _amp_to_db(np.abs(D)) - hparams['ref_level_db']
    res = _normalize(S, hparams)

    # logging.info ('spectrogram y:%s D:%s S:%s res:%s' % (y.shape, D.shape, S.shape, res.shape))

    return res

def melspectrogram(y, hparams):
    n_fft, hop_length, win_length = stft_parameters(hparams)
    D = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams)) - hparams['ref_level_db']
    return _normalize(S, hparams)

def find_endpoint(wav, hparams, threshold_db=-40, min_silence_sec=0.5):
    window_length = int(hparams['sample_rate'] * min_silence_sec)
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


def pad_center(data, size, axis=-1, **kwargs):

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        import pdb; pdb.set_trace()
        raise Exception(('Target size ({:d}) must be at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)

def _valid_audio(y, mono=True):

    if not isinstance(y, np.ndarray):
        raise Exception('data must be of type numpy.ndarray')

    if not np.issubdtype(y.dtype, np.floating):
        raise Exception('data must be floating-point')

    if mono and y.ndim != 1:
        raise Exception('Invalid shape for monophonic audio: '
                             'ndim={:d}, shape={}'.format(y.ndim, y.shape))

    elif y.ndim > 2 or y.ndim == 0:
        raise Exception('Audio must have shape (samples,) or (channels, samples). '
                             'Received shape={}'.format(y.shape))

    if not np.isfinite(y).all():
        raise Exception('Audio buffer is not finite everywhere')

    return True

def _frame(y, frame_length=2048, hop_length=512):
    if not isinstance(y, np.ndarray):
        raise Exception('Input must be of type numpy.ndarray, '
                             'given type(y)={}'.format(type(y)))

    if y.ndim != 1:
        raise Exception('Input must be one-dimensional, '
                             'given y.ndim={}'.format(y.ndim))

    if len(y) < frame_length:
        raise Exception('Buffer is too short (n={:d})'
                             ' for frame_length={:d}'.format(len(y), frame_length))

    if hop_length < 1:
        raise Exception('Invalid hop_length: {:d}'.format(hop_length))

    if not y.flags['C_CONTIGUOUS']:
        raise Exception('Input buffer must be contiguous.')

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

        raise Exception('Window size mismatch: '
                             '{:d} != {:d}'.format(len(window), Nx))
    else:
        raise Exception('Invalid window specification: {}'.format(window))

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
        raise Exception('threshold={} must be strictly '
                             'positive'.format(threshold))

    if fill not in [None, False, True]:
        raise Exception('fill={} must be None or boolean'.format(fill))

    if not np.all(np.isfinite(S)):
        raise Exception('Input must be finite')

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
            raise Exception('Cannot normalize with norm=0 and fill=True')

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
        raise Exception('Unsupported norm: {}'.format(repr(norm)))

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


_mel_basis = None

def _linear_to_mel(spectrogram, hparams):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis(hparams)
  return np.dot(_mel_basis, spectrogram)

def _fft_frequencies(sr=22050, n_fft=2048):
    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)

def hz_to_mel(frequencies, htk=False):

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs

def _mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def _mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1):
    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise Exception('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = _fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = _mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights

def _build_mel_basis(hparams):
    n_fft = (hparams['num_freq'] - 1) * 2
    return _mel(hparams['sample_rate'], n_fft, n_mels=hparams['num_mels'],
                fmin=hparams['min_mel_freq'], fmax=hparams['max_mel_freq'])

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _normalize(S, hparams):
    return np.clip((S - hparams['min_level_db']) / -hparams['min_level_db'], 0, 1)

def _denormalize(S, hparams):
    return (np.clip(S, 0, 1) * -hparams['min_level_db']) + hparams['min_level_db']

def load_wav(wavfn):

    wavf = wave.open(wavfn, 'rb')

    # check format

    if wavf.getnchannels() != 1:
        raise Exception ('mono wav needed.')
    if wavf.getsampwidth() != 2:
        raise Exception ('16 bit wav needed.')
    if wavf.getframerate() != 16000:
        raise Exception ('16 kHz wav needed.')
    
    # read the whole file

    tot_frames   = wavf.getnframes()
    frames = wavf.readframes(tot_frames)

    # print 'frames:', repr(frames)

    samples = struct.unpack_from('<%dh' % tot_frames, frames)

    # print 'samples:', type(samples)

    wavf.close()

    return np.asarray(samples, dtype=np.float32) / 32768.0

def save_wav(wav, wavfn, hparams):

    wav16 = (32767 * wav).astype(np.int16)

    scipy.io.wavfile.write(wavfn, hparams['sample_rate'], wav16)

def _to_mono(y):

    # Validate the buffer.  Stereo is ok here.
    _valid_audio(y, mono=False)

    if y.ndim > 1:
        y = np.mean(y, axis=0)

    return y

def _rmse(y, frame_length=2048, hop_length=512,
          center=True, pad_mode='reflect'):

    if center:
        y = np.pad(y, int(frame_length // 2), mode=pad_mode)

    x = _frame(y,
               frame_length=frame_length,
               hop_length=hop_length)

    return np.sqrt(np.mean(np.abs(x)**2, axis=0, keepdims=True))

def _power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):

    S = np.asarray(S)

    if amin <= 0:
        raise Exception('amin must be strictly positive')

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('power_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call power_to_db(np.abs(D)**2) instead.')
        magnitude = np.abs(S)
    else:
        magnitude = S

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise Exception('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec

def _signal_to_frame_nonsilent(y, frame_length=2048, hop_length=512, top_db=60, ref=np.max):

    # Convert to mono
    y_mono = _to_mono(y)

    # Compute the MSE for the signal
    mse = _rmse(y_mono,
                frame_length=frame_length,
                hop_length=hop_length)**2

    return _power_to_db(mse.squeeze(), ref=ref, top_db=None) > -top_db

def _frames_to_samples(frames, hop_length=512, n_fft=None):
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def _trim(y, top_db=60, ref=np.max, frame_length=2048, hop_length=512):
    non_silent = _signal_to_frame_nonsilent(y,
                                            frame_length=frame_length,
                                            hop_length=hop_length,
                                            ref=ref,
                                            top_db=top_db)

    nonzero = np.flatnonzero(non_silent)

    if nonzero.size > 0:
        # Compute the start and end positions
        # End position goes one frame past the last non-zero
        start = int(_frames_to_samples(nonzero[0], hop_length))
        end = min(y.shape[-1], int(_frames_to_samples(nonzero[-1] + 1, hop_length)))
    else:
        # The signal only contains zeros
        start, end = 0, 0

    # Build the mono/stereo index
    full_index = [slice(None)] * y.ndim
    full_index[-1] = slice(start, end)

    return y[tuple(full_index)], np.asarray([start, end])


def trim_silence(wav, hparams):
    return _trim( wav, top_db=hparams['trim_top_db'], frame_length=hparams['trim_fft_size'], hop_length=hparams['trim_hop_size'])[0]


#########################################################################
# 
# alternative, possibly faster implementation
#
#########################################################################


def _spsi(msgram, fftsize, hop_length, center=True) :
    """
    Takes a 2D spectrogram ([freqs,frames]), the fft legnth (= window length) and the hop size (both in units of samples).
    Returns an audio signal.

    Single Pass Spectrogram Inversion
    Reconstruct an audio signal from a magnitude spectrum with but a single ifft.
    
    Cite and more info: 
    Beauregard, G., Harish, M. and Wyse, L. (2015), Single Pass Spectrogram Inversion, 
    in Proceedings of the IEEE International Conference on Digital Signal Processing. Singapore, 2015.

    source: https://github.com/lonce/SPSI_Python/blob/master/SPSI_notebook/spsi.ipynb
    """
    
    numBins, numFrames  = msgram.shape

    # y_out=np.zeros(numFrames*hop_length+fftsize-hop_length)

    expected_signal_len = fftsize + hop_length * (numFrames - 1)
    # print '_spsi: expected_signal_len:', expected_signal_len
    y_out=np.zeros(expected_signal_len)
        
    m_phase=np.zeros(numBins);      
    m_win=scipy.signal.hanning(fftsize, sym=True)  # assumption here that hann was used to create the frames of the spectrogram
    
    #processes one frame of audio at a time
    for i in range(numFrames) :
        m_mag=msgram[:, i] 
        for j in range(1,numBins-1) : 
            if(m_mag[j]>m_mag[j-1] and m_mag[j]>m_mag[j+1]) : #if j is a peak
                alpha=m_mag[j-1];
                beta=m_mag[j];
                gamma=m_mag[j+1];
                denom=alpha-2*beta+gamma;
                
                if(denom!=0) :
                    p=0.5*(alpha-gamma)/denom;
                else :
                    p=0;
                    
                #phaseRate=2*math.pi*(j-1+p)/fftsize;    #adjusted phase rate
                phaseRate=2*math.pi*(j+p)/fftsize;    #adjusted phase rate
                m_phase[j]= m_phase[j] + hop_length*phaseRate; #phase accumulator for this peak bin
                peakPhase=m_phase[j];
                
                # If actual peak is to the right of the bin freq
                if (p>0) :
                    # First bin to right has pi shift
                    bin=j+1;
                    m_phase[bin]=peakPhase+math.pi;
                    
                    # Bins to left have shift of pi
                    bin=j-1;
                    while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until you reach the trough
                        m_phase[bin]=peakPhase+math.pi;
                        bin=bin-1;
                    
                    #Bins to the right (beyond the first) have 0 shift
                    bin=j+2;
                    while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                        m_phase[bin]=peakPhase;
                        bin=bin+1;
                        
                #if actual peak is to the left of the bin frequency
                if(p<0) :
                    # First bin to left has pi shift
                    bin=j-1;
                    m_phase[bin]=peakPhase+math.pi;

                    # and bins to the right of me - here I am stuck in the middle with you
                    bin=j+1;
                    while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                        m_phase[bin]=peakPhase+math.pi;
                        bin=bin+1;
                    
                    # and further to the left have zero shift
                    bin=j-2;
                    while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until trough
                        m_phase[bin]=peakPhase;
                        bin=bin-1;
                        
            #end ops for peaks
        #end loop over fft bins with

        magphase=m_mag*np.exp(1j*m_phase)  #reconstruct with new phase (elementwise mult)
        magphase[0]=0; magphase[numBins-1] = 0 #remove dc and nyquist
        m_recon=np.concatenate([magphase,np.flip(np.conjugate(magphase[1:numBins-1]), 0)]) 
        
        #overlap and add
        m_recon=np.real(np.fft.ifft(m_recon))*m_win
        y_out[i*hop_length:i*hop_length+fftsize]+=m_recon
            
    if center:
        y_out = y_out[int(fftsize // 2):-int(fftsize // 2)]

    return y_out

def inv_spectrogram_spsi(spectrogram, hparams):
    S = _db_to_amp(_denormalize(spectrogram) + hparams['ref_level_db'])  # Convert back to linear
    n_fft, hop_length, win_length = stft_parameters(hparams)
    return _spsi(S ** hparams['power'], n_fft, hop_length)

def inv_spectrogram_spsi2(spectrogram, griffin_lim_iters):
    S = _db_to_amp(_denormalize(spectrogram) + hparams['ref_level_db'])  # Convert back to linear
    S = S ** hparams['power']

    # print 'inv_spectrogram_spsi2: S.shape:', S.shape

    n_fft, hop_length, win_length = stft_parameters(hparams)
    y = _spsi(S, n_fft, hop_length)

    # print 'inv_spectrogram_spsi2: y.shape:', y.shape

    S_complex = np.abs(S).astype(np.complex)

    # print 'inv_spectrogram_spsi2: S.shape:', S.shape
    # print 'inv_spectrogram_spsi2: S_complex.shape:', S_complex.shape

    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)))
        # print 'inv_spectrogram_spsi2: angles.shape:', angles.shape
        y = istft(S_complex * angles, hop_length=hop_length, win_length=win_length)

    return y

