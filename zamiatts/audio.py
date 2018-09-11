# import librosa
# import librosa.filters
import math
import numpy as np
import tensorflow as tf
from hparams import hparams

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


def inv_spectrogram_tensorflow(spectrogram):
  '''Builds computational graph to convert spectrogram to waveform using TensorFlow.'''
  S = _db_to_amp_tensorflow(_denormalize_tensorflow(
      spectrogram) + hparams.ref_level_db)
  return _griffin_lim_tensorflow(tf.pow(S, hparams.power))


def melspectrogram(y):
  D = _stft(y)
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
  return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hparams.sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x + window_length]) < threshold:
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


def _griffin_lim_tensorflow(S):
  '''TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
  '''
  with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(hparams.griffin_lim_iters):
      est = _stft_tensorflow(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)

# def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
#          center=True, dtype=np.complex64, pad_mode='reflect'):
# 
#     # By default, use the entire frame
#     if win_length is None:
#         win_length = n_fft
# 
#     # Set the default hop, if it's not already specified
#     if hop_length is None:
#         hop_length = int(win_length // 4)
# 
#     fft_window = get_window(window, win_length, fftbins=True)
# 
#     # Pad the window out to n_fft size
#     fft_window = util.pad_center(fft_window, n_fft)
# 
#     # Reshape so that the window can be broadcast
#     fft_window = fft_window.reshape((-1, 1))
# 
#     # Check audio is valid
#     util.valid_audio(y)
# 
#     # Pad the time series so that frames are centered
#     if center:
#         y = np.pad(y, int(n_fft // 2), mode=pad_mode)
# 
#     # Window the time series.
#     y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)
# 
#     # Pre-allocate the STFT matrix
#     stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
#                            dtype=dtype,
#                            order='F')
# 
#     # how many columns can we fit within MAX_MEM_BLOCK?
#     n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
#                                           stft_matrix.itemsize))
# 
#     for bl_s in range(0, stft_matrix.shape[1], n_columns):
#         bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
# 
#         # RFFT and Conjugate here to match phase from DPWE code
#         stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
#                                             y_frames[:, bl_s:bl_t],
#                                             axis=0)[:stft_matrix.shape[0]]
# 
#     return stft_matrix
# 
# 
# def _stft(y):
#   n_fft, hop_length, win_length = _stft_parameters()
#   return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
# 
# def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
#           center=True, dtype=np.float32, length=None):
#     n_fft = 2 * (stft_matrix.shape[0] - 1)
# 
#     # By default, use the entire frame
#     if win_length is None:
#         win_length = n_fft
# 
#     # Set the default hop, if it's not already specified
#     if hop_length is None:
#         hop_length = int(win_length // 4)
# 
#     ifft_window = get_window(window, win_length, fftbins=True)
# 
#     # Pad out to match n_fft
#     ifft_window = util.pad_center(ifft_window, n_fft)
# 
#     n_frames = stft_matrix.shape[1]
#     expected_signal_len = n_fft + hop_length * (n_frames - 1)
#     y = np.zeros(expected_signal_len, dtype=dtype)
# 
#     for i in range(n_frames):
#         sample = i * hop_length
#         spec = stft_matrix[:, i].flatten()
#         spec = np.concatenate((spec, spec[-2:0:-1].conj()), 0)
#         ytmp = ifft_window * fft.ifft(spec).real
# 
#         y[sample:(sample + n_fft)] = y[sample:(sample + n_fft)] + ytmp
# 
#     # Normalize by sum of squared window
#     ifft_window_sum = window_sumsquare(window,
#                                        n_frames,
#                                        win_length=win_length,
#                                        n_fft=n_fft,
#                                        hop_length=hop_length,
#                                        dtype=dtype)
# 
#     approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)
#     y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]
# 
#     if length is None:
#         # If we don't need to control length, just do the usual center trimming
#         # to eliminate padded data
#         if center:
#             y = y[int(n_fft // 2):-int(n_fft // 2)]
#     else:
#         if center:
#             # If we're centering, crop off the first n_fft//2 samples
#             # and then trim/pad to the target length.
#             # We don't trim the end here, so that if the signal is zero-padded
#             # to a longer duration, the decay is smooth by windowing
#             start = int(n_fft // 2)
#         else:
#             # If we're not centering, start at 0 and trim/pad as necessary
#             start = 0
# 
#         y = util.fix_length(y[start:], length)
# 
#     return y
# 
# 
# def _istft(y):
#   _, hop_length, win_length = _stft_parameters()
#   return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(stfts):
  n_fft, hop_length, win_length = _stft_parameters()
  print stfts, win_length, hop_length, n_fft
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


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


def _denormalize_tensorflow(S):
  return (tf.clip_by_value(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
