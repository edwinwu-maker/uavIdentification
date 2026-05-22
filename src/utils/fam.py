from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np


WindowType = Literal["hann", "hamming", "rect"]
NormalizeType = Literal["none", "zscore", "minmax"]


def _validate_1d_iq(sig_iq: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig_iq)
    if sig.ndim != 1:
        sig = sig.reshape(-1)
    if sig.size == 0:
        raise ValueError("sig_iq must contain at least one sample")
    if not np.iscomplexobj(sig):
        sig = sig.astype(np.float32) + 0j
    return sig.astype(np.complex64, copy=False)


def _make_window(window: WindowType, frame_len: int) -> np.ndarray:
    if window == "hann":
        return np.hanning(frame_len).astype(np.float32)
    if window == "hamming":
        return np.hamming(frame_len).astype(np.float32)
    if window == "rect":
        return np.ones(frame_len, dtype=np.float32)
    raise ValueError(f"Unsupported window type: {window}")


def _frame_signal(sig: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if frame_len <= 0:
        raise ValueError("frame_len must be positive")
    if hop_len <= 0:
        raise ValueError("hop_len must be positive")
    if sig.size < frame_len:
        raise ValueError(
            f"sig_iq is too short: got {sig.size} samples, need at least {frame_len}"
        )

    num_frames = 1 + (sig.size - frame_len) // hop_len
    shape = (num_frames, frame_len)
    strides = (sig.strides[0] * hop_len, sig.strides[0])
    return np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides)


def _select_evenly(values: np.ndarray, count: Optional[int]) -> np.ndarray:
    if count is None or count >= values.size:
        return values
    if count <= 0:
        raise ValueError("freq_bins must be positive when provided")
    indices = np.linspace(0, values.size - 1, count).round().astype(np.int64)
    return values[indices]


def _normalize_spectrum(
    spectrum: np.ndarray,
    mode: NormalizeType,
    eps: float,
) -> np.ndarray:
    if mode == "none":
        return spectrum
    if mode == "zscore":
        return (spectrum - spectrum.mean()) / (spectrum.std() + eps)
    if mode == "minmax":
        min_val = spectrum.min()
        max_val = spectrum.max()
        return (spectrum - min_val) / (max_val - min_val + eps)
    raise ValueError(f"Unsupported normalize mode: {mode}")


def estimate_fam_cyclic_spectrum(
    sig_iq: np.ndarray,
    sample_rate: float = 1.0,
    frame_len: int = 1024,
    overlap: float = 0.5,
    fft_len: Optional[int] = None,
    alpha_bins: int = 128,
    freq_bins: Optional[int] = 256,
    alpha_max: Optional[float] = None,
    include_alpha_zero: bool = False,
    window: WindowType = "hann",
    remove_dc: bool = True,
    power_normalize: bool = True,
    log_scale: bool = True,
    normalize: NormalizeType = "zscore",
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a second-order cyclic spectrum with an FAM-style approximation.

    Parameters
    ----------
    sig_iq:
        One-dimensional complex IQ signal.
    sample_rate:
        Sampling rate in Hz. Use 1.0 for normalized frequency axes.
    frame_len:
        Number of IQ samples per short-time FFT frame.
    overlap:
        Frame overlap ratio in [0, 1).
    fft_len:
        FFT length. Defaults to ``frame_len``.
    alpha_bins:
        Number of cyclic-frequency bins requested. The actual count can be
        lower if the integer FFT-bin grid cannot provide enough unique shifts.
    freq_bins:
        Number of frequency bins returned. ``None`` returns all valid bins.
    alpha_max:
        Maximum cyclic frequency in Hz. Defaults to ``sample_rate / 2``.
    include_alpha_zero:
        Whether to include alpha = 0, which corresponds to ordinary PSD-like
        spectral correlation.
    window:
        Short-time analysis window.
    remove_dc:
        Remove the IQ mean before framing.
    power_normalize:
        Normalize IQ average power to 1 before FAM estimation.
    log_scale:
        Apply ``log1p(abs(.))`` to the spectral correlation magnitude.
    normalize:
        Output normalization mode: ``none``, ``zscore``, or ``minmax``.
    eps:
        Small positive constant for numerical stability.

    Returns
    -------
    spectrum:
        Float32 array with shape ``(N_alpha, N_freq)``.
    alpha_axis:
        Cyclic-frequency axis in Hz, shape ``(N_alpha,)``.
    freq_axis:
        Frequency axis in Hz, shape ``(N_freq,)``.

    Notes
    -----
    This implementation uses integer FFT-bin shifts:

    ``S_alpha(f) = mean_t X_t(f + alpha / 2) * conj(X_t(f - alpha / 2))``

    Therefore alpha values are quantized to ``2 * delta_bin * Fs / fft_len``.
    It is intended as a clear CPU baseline for offline feature precomputation.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if alpha_bins <= 0:
        raise ValueError("alpha_bins must be positive")

    sig = _validate_1d_iq(sig_iq)
    if remove_dc:
        sig = sig - sig.mean()
    if power_normalize:
        sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2) + eps)

    if fft_len is None:
        fft_len = frame_len
    if fft_len < frame_len:
        raise ValueError("fft_len must be greater than or equal to frame_len")

    hop_len = max(1, int(round(frame_len * (1.0 - overlap))))
    frames = _frame_signal(sig, frame_len=frame_len, hop_len=hop_len)
    frames = frames * _make_window(window, frame_len)[None, :]

    stft = np.fft.fft(frames, n=fft_len, axis=1)
    stft = np.fft.fftshift(stft, axes=1)

    if alpha_max is None:
        alpha_max = sample_rate / 2.0
    if alpha_max <= 0:
        raise ValueError("alpha_max must be positive")

    max_delta = int(np.floor(alpha_max * fft_len / (2.0 * sample_rate)))
    max_delta = min(max_delta, (fft_len - 1) // 2)
    min_delta = 0 if include_alpha_zero else 1
    if max_delta < min_delta:
        raise ValueError("alpha_max is too small for the requested FFT grid")

    requested = min(alpha_bins, max_delta - min_delta + 1)
    deltas = np.linspace(min_delta, max_delta, requested).round().astype(np.int64)
    deltas = np.unique(deltas)

    full_freq_axis = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1.0 / sample_rate))
    valid_freq_indices = np.arange(max_delta, fft_len - max_delta, dtype=np.int64)
    freq_indices = _select_evenly(valid_freq_indices, freq_bins)

    spectrum = np.empty((deltas.size, freq_indices.size), dtype=np.float32)
    for row, delta in enumerate(deltas):
        upper = stft[:, freq_indices + delta]
        lower = stft[:, freq_indices - delta]
        corr = np.mean(upper * np.conj(lower), axis=0)
        mag = np.abs(corr)
        if log_scale:
            mag = np.log1p(mag)
        spectrum[row] = mag.astype(np.float32, copy=False)

    spectrum = _normalize_spectrum(spectrum, normalize, eps).astype(np.float32, copy=False)
    alpha_axis = (2.0 * deltas * sample_rate / fft_len).astype(np.float32)
    freq_axis = full_freq_axis[freq_indices].astype(np.float32)

    return spectrum, alpha_axis, freq_axis


def estimate_fam_image(
    sig_iq: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Return only the cyclic-spectrum image for dataset precomputation."""
    spectrum, _alpha_axis, _freq_axis = estimate_fam_cyclic_spectrum(sig_iq, **kwargs)
    return spectrum
