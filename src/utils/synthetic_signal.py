"""Synthetic IQ signal generators used by FAM visualization scripts."""

from __future__ import annotations

import numpy as np

SUPPORTED_SIGNAL_TYPES = ("bpsk", "bpsk_noise", "noise")
DEFAULT_SAMPLES_PER_SYMBOL = 10


def generate_bpsk(
    *,
    num_symbols: int = 1024,
    samples_per_symbol: int = DEFAULT_SAMPLES_PER_SYMBOL,
    rolloff: float = 0.35,
    span_symbols: int = 8,
    seed: int = 7,
) -> np.ndarray:
    """Generate oversampled BPSK IQ after root-raised-cosine pulse shaping."""

    rng = np.random.default_rng(seed)
    symbols = 2 * rng.integers(0, 2, size=num_symbols) - 1

    upsampled = np.zeros(num_symbols * samples_per_symbol, dtype=float)
    upsampled[::samples_per_symbol] = symbols
    taps = root_raised_cosine(
        samples_per_symbol=samples_per_symbol,
        rolloff=rolloff,
        span_symbols=span_symbols,
    )
    shaped = np.convolve(upsampled, taps, mode="same")
    return shaped.astype(np.complex128)


def generate_awgn(
    *,
    num_samples: int,
    power: float = 1.0,
    seed: int = 7,
) -> np.ndarray:
    """Generate complex additive white Gaussian noise with the requested power."""

    rng = np.random.default_rng(seed)
    sigma = np.sqrt(power / 2.0)
    noise = sigma * (
        rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)
    )
    return noise.astype(np.complex128)


def add_awgn_for_snr(
    x: np.ndarray,
    *,
    snr_db: float,
    seed: int = 7,
) -> np.ndarray:
    """Add complex AWGN to make the output match the requested SNR in dB."""

    signal_power = float(np.mean(np.abs(x) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = generate_awgn(num_samples=len(x), power=noise_power, seed=seed)
    return x + noise


def generate_signal(
    signal_type: str,
    *,
    num_symbols: int = 200000,
    samples_per_symbol: int = DEFAULT_SAMPLES_PER_SYMBOL,
    snr_db: float = 10.0,
    seed: int = 7,
) -> np.ndarray:
    """Generate the selected IQ signal for FAM visualization."""

    if signal_type == "bpsk":
        return generate_bpsk(
            num_symbols=num_symbols,
            samples_per_symbol=samples_per_symbol,
            seed=seed,
        )
    if signal_type == "bpsk_noise":
        bpsk = generate_bpsk(
            num_symbols=num_symbols,
            samples_per_symbol=samples_per_symbol,
            seed=seed,
        )
        return add_awgn_for_snr(bpsk, snr_db=snr_db, seed=seed + 1)
    if signal_type == "noise":
        return generate_awgn(
            num_samples=num_symbols * samples_per_symbol,
            seed=seed,
        )
    raise ValueError(f"unsupported signal type: {signal_type}")


def root_raised_cosine(
    *,
    samples_per_symbol: int,
    rolloff: float,
    span_symbols: int,
) -> np.ndarray:
    """Return energy-normalized root-raised-cosine FIR taps."""

    half_len = span_symbols * samples_per_symbol // 2
    t = np.arange(-half_len, half_len + 1, dtype=float) / samples_per_symbol
    taps = np.empty_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 + rolloff * (4.0 / np.pi - 1.0)
        elif rolloff > 0 and np.isclose(abs(ti), 1.0 / (4.0 * rolloff)):
            taps[i] = (
                rolloff
                / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * rolloff))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * rolloff))
                )
            )
        else:
            numerator = (
                np.sin(np.pi * ti * (1.0 - rolloff))
                + 4.0 * rolloff * ti * np.cos(np.pi * ti * (1.0 + rolloff))
            )
            denominator = np.pi * ti * (1.0 - (4.0 * rolloff * ti) ** 2)
            taps[i] = numerator / denominator

    return taps / np.sqrt(np.sum(taps**2))
