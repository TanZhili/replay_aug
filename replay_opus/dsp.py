"""Low-level signal operations shared by every stage of the replay chain."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

EPS = 1e-12


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x
    axis = 0 if x.shape[0] < x.shape[1] else 1
    return x.mean(axis=axis)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + EPS))


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(rms(x) + EPS)


def peak_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(float(np.max(np.abs(x))) + EPS)


def set_peak_db(x: np.ndarray, target_db: float) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    if peak < EPS:
        return x
    return x * (10.0 ** (target_db / 20.0) / peak)


def butter(
    x: np.ndarray,
    sr: int,
    cutoff: float | tuple[float, float],
    kind: str = "high",
    order: int = 4,
) -> np.ndarray:
    nyq = sr / 2.0
    if isinstance(cutoff, tuple):
        wn = (min(cutoff[0] / nyq, 0.999), min(cutoff[1] / nyq, 0.999))
    else:
        wn = min(cutoff / nyq, 0.999)
    sos = signal.butter(order, wn, btype=kind, output="sos")
    return signal.sosfilt(sos, x)


def peaking_eq(x: np.ndarray, sr: int, f0: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """RBJ peaking filter. Used to give each simulated device its own spectral tilt."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b = np.array([1 + alpha * a, -2 * cos_w0, 1 - alpha * a])
    a_coef = np.array([1 + alpha / a, -2 * cos_w0, 1 - alpha / a])
    return signal.lfilter(b / a_coef[0], a_coef / a_coef[0], x)


def random_spectral_tilt(
    x: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    n_bands: int = 4,
    max_gain_db: float = 6.0,
) -> tuple[np.ndarray, list[dict]]:
    """Apply a few random peaking filters so no two simulated devices sound alike."""
    applied: list[dict] = []
    y = x
    for _ in range(n_bands):
        f0 = float(np.exp(rng.uniform(np.log(120.0), np.log(min(7000.0, sr / 2 * 0.9)))))
        gain = float(rng.uniform(-max_gain_db, max_gain_db))
        q = float(rng.uniform(0.5, 2.0))
        y = peaking_eq(y, sr, f0, gain, q)
        applied.append({"f0_hz": round(f0, 1), "gain_db": round(gain, 2), "q": round(q, 2)})
    return y, applied


def soft_clip(x: np.ndarray, kind: str = "tanh", drive: float = 1.0) -> np.ndarray:
    if kind == "none":
        return x
    if kind == "hard":
        return np.clip(x * drive, -1.0, 1.0)
    if kind == "cubic":
        v = np.clip(x * drive, -1.0, 1.0)
        return 1.5 * v - 0.5 * v**3
    return np.tanh(x * drive)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling at the exact rate ratio where that is cheap.

    Standard audio rates reduce to small fractions (44.1k to 48k is 160/147), but an
    unusual pair can reduce to one with a huge denominator, which would make the
    polyphase filter enormous. Those fall back to an approximation.
    """
    if sr_in == sr_out:
        return x
    ratio = Fraction(sr_out, sr_in)
    if max(ratio.numerator, ratio.denominator) > 2000:
        ratio = ratio.limit_denominator(2000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator)


def apply_clock_drift(x: np.ndarray, ppm: float) -> np.ndarray:
    """Playback and capture crystals never match exactly; that is a tiny resample.

    Band-limited (FFT) resampling is used because the offsets of interest are far
    below one sample per block, where drop/duplicate strategies would be audible.
    """
    if abs(ppm) < 1e-6:
        return x
    n_out = round(len(x) * (1.0 + ppm * 1e-6))
    if n_out == len(x) or n_out < 8:
        return x
    return signal.resample(x, n_out)


def align_to(
    y: np.ndarray, ref: np.ndarray, max_lag: int = 8192, max_samples: int = 200_000
) -> tuple[np.ndarray, int]:
    """Remove a bulk time shift from ``y`` relative to ``ref``.

    Codecs emit priming/padding samples, so a decoded signal comes back offset by a
    codec-dependent number of samples. Left uncorrected that becomes a random latency
    baked into every generated file. Only the first ``max_samples`` are correlated,
    which bounds the cost on long files.
    """
    n = min(len(y), len(ref), max_samples)
    if n < 64:
        return y, 0
    corr = signal.correlate(y[:n] - np.mean(y[:n]), ref[:n] - np.mean(ref[:n]), mode="full")
    lags = signal.correlation_lags(n, n, mode="full")
    keep = np.abs(lags) <= max_lag
    lag = int(lags[keep][np.argmax(corr[keep])])
    if lag > 0:
        return y[lag:], lag
    if lag < 0:
        return np.pad(y, (-lag, 0)), lag
    return y, 0


def fade_edges(x: np.ndarray, sr: int, ms: float = 5.0) -> np.ndarray:
    n = max(1, int(sr * ms / 1000.0))
    if len(x) < 2 * n:
        return x
    ramp = np.hanning(2 * n)
    y = x.copy()
    y[:n] *= ramp[:n]
    y[-n:] *= ramp[n:]
    return y


def sample_range(rng: np.random.Generator, spec, log: bool = False) -> float:
    """Draw from ``[lo, hi]`` config ranges; scalars pass through unchanged."""
    if isinstance(spec, (int, float)):
        return float(spec)
    lo, hi = float(spec[0]), float(spec[1])
    if lo == hi:
        return lo
    if log:
        return float(np.exp(rng.uniform(np.log(max(lo, EPS)), np.log(hi))))
    return float(rng.uniform(lo, hi))
