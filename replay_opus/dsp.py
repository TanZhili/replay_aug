"""Low-level signal operations shared by every stage.

Filters are designed as second-order sections so that a whole cascade (a loudspeaker's
high-pass, resonance, ripples and roll-off) runs in a single ``sosfilt`` call.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import signal

EPS = 1e-12
P_REF = 20e-6  # reference sound pressure for dB SPL, in pascals


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x
    axis = 0 if x.shape[0] < x.shape[1] else 1
    return x.mean(axis=axis)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + EPS))


# ---------------------------------------------------------------------------
# Filter design. Everything returns an (n, 6) SOS array.
# ---------------------------------------------------------------------------

EMPTY_SOS = np.zeros((0, 6))


def butter_sos(
    sr: int, cutoff: float | tuple[float, float], kind: str = "high", order: int = 4
) -> np.ndarray:
    nyq = sr / 2.0
    if isinstance(cutoff, tuple):
        wn: float | tuple[float, float] = (
            min(cutoff[0] / nyq, 0.999),
            min(cutoff[1] / nyq, 0.999),
        )
    else:
        wn = min(cutoff / nyq, 0.999)
    return signal.butter(int(order), wn, btype=kind, output="sos")


def linkwitz_riley_sos(sr: int, cutoff: float, kind: str, order: int = 4) -> np.ndarray:
    """Two cascaded Butterworths: the low and high outputs sum to an all-pass."""
    half = butter_sos(sr, cutoff, kind, max(1, order // 2))
    return np.vstack([half, half])


def peaking_sos(sr: int, f0: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """RBJ peaking biquad. Returns an empty cascade for centres too close to Nyquist."""
    if f0 >= 0.45 * sr or abs(gain_db) < 1e-3:
        return EMPTY_SOS
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b = np.array([1 + alpha * a, -2 * cos_w0, 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * cos_w0, 1 - alpha / a])
    return np.concatenate([b / den[0], den / den[0]])[None, :]


def shelf_sos(
    sr: int, f0: float, gain_db: float, kind: str = "high", q: float = 0.707
) -> np.ndarray:
    """RBJ shelving biquad."""
    if f0 >= 0.45 * sr or abs(gain_db) < 1e-3:
        return EMPTY_SOS
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    root = 2.0 * np.sqrt(a) * alpha
    if kind == "high":
        b = a * np.array(
            [
                (a + 1) + (a - 1) * cos_w0 + root,
                -2 * ((a - 1) + (a + 1) * cos_w0),
                (a + 1) + (a - 1) * cos_w0 - root,
            ]
        )
        den = np.array(
            [
                (a + 1) - (a - 1) * cos_w0 + root,
                2 * ((a - 1) - (a + 1) * cos_w0),
                (a + 1) - (a - 1) * cos_w0 - root,
            ]
        )
    else:
        b = a * np.array(
            [
                (a + 1) - (a - 1) * cos_w0 + root,
                2 * ((a - 1) - (a + 1) * cos_w0),
                (a + 1) - (a - 1) * cos_w0 - root,
            ]
        )
        den = np.array(
            [
                (a + 1) + (a - 1) * cos_w0 + root,
                -2 * ((a - 1) + (a + 1) * cos_w0),
                (a + 1) + (a - 1) * cos_w0 - root,
            ]
        )
    return np.concatenate([b / den[0], den / den[0]])[None, :]


def cascade(*parts: np.ndarray) -> np.ndarray:
    kept = [p for p in parts if p is not None and len(p)]
    return np.vstack(kept) if kept else EMPTY_SOS


def apply_sos(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    if sos is None or len(sos) == 0:
        return x
    return signal.sosfilt(sos, x)


def butter(
    x: np.ndarray,
    sr: int,
    cutoff: float | tuple[float, float],
    kind: str = "high",
    order: int = 4,
) -> np.ndarray:
    return apply_sos(x, butter_sos(sr, cutoff, kind, order))


def sos_response_db(sos: np.ndarray, sr: int, freqs: np.ndarray) -> np.ndarray:
    if len(sos) == 0:
        return np.zeros_like(freqs, dtype=np.float64)
    _, h = signal.sosfreqz(sos, worN=np.asarray(freqs, dtype=np.float64), fs=sr)
    return 20.0 * np.log10(np.abs(h) + EPS)


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def a_weighting_sos(sr: int) -> np.ndarray:
    """IEC 61672 A-weighting, bilinear-transformed and pinned to 0 dB at 1 kHz."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    two_pi = 2.0 * np.pi
    zeros = np.zeros(4)
    poles = -two_pi * np.array([f1, f1, f2, f3, f4, f4])
    z, p, k = signal.bilinear_zpk(zeros, poles, (two_pi * f4) ** 2, sr)
    sos = signal.zpk2sos(z, p, k)
    gain_1k = sos_response_db(sos, sr, np.array([1000.0]))[0]
    sos[0, :3] *= 10.0 ** (-gain_1k / 20.0)
    return sos


def a_weight(x: np.ndarray, sr: int) -> np.ndarray:
    return signal.sosfilt(a_weighting_sos(sr), x)


def level_dba(x_pa: np.ndarray, sr: int) -> float:
    """A-weighted sound pressure level of a signal expressed in pascals."""
    return 20.0 * np.log10(rms(a_weight(x_pa, sr)) / P_REF + EPS)


def frame_energy(x: np.ndarray, hop: int) -> np.ndarray:
    n_frames = len(x) // hop
    if n_frames < 1:
        return np.array([float(np.mean(np.square(x))) if len(x) else 0.0])
    return np.mean(np.square(x[: n_frames * hop]).reshape(n_frames, hop), axis=1)


def active_mask(
    x: np.ndarray, sr: int, frame_ms: float = 20.0, threshold_db: float = -35.0
) -> tuple[np.ndarray, int]:
    """Frames within ``threshold_db`` of the loudest one. Returns ``(mask, hop)``."""
    hop = max(1, int(sr * frame_ms / 1000.0))
    energy = frame_energy(x, hop)
    energy_db = 10.0 * np.log10(energy + EPS)
    return energy_db > energy_db.max() + threshold_db, hop


def active_level_db(
    x: np.ndarray,
    sr: int,
    weighted: bool = True,
    frame_ms: float = 20.0,
    threshold_db: float = -35.0,
) -> float:
    """Level over the active frames only, A-weighted by default.

    A simplified take on the ITU-T P.56 active speech level. TTS output carries long
    leading and trailing silences, and a plain RMS would make every file's level an
    accident of how much silence it happens to contain.
    """
    y = a_weight(x, sr) if weighted else np.asarray(x, dtype=np.float64)
    hop = max(1, int(sr * frame_ms / 1000.0))
    energy = frame_energy(y, hop)
    energy_db = 10.0 * np.log10(energy + EPS)
    active = energy_db > energy_db.max() + threshold_db
    return float(10.0 * np.log10(np.mean(energy[active]) + EPS))


def speech_segments(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -35.0,
    frame_ms: float = 10.0,
    min_gap_s: float = 0.2,
    min_dur_s: float = 0.06,
) -> list[tuple[float, float]]:
    """Voice-active regions of a clean signal, in seconds.

    Meant for dry TTS, where an energy threshold relative to the loudest frame is
    reliable. Gaps shorter than ``min_gap_s`` are bridged so a label covers a phrase
    rather than every syllable.
    """
    mask, hop = active_mask(x, sr, frame_ms, threshold_db)
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    runs = list(zip(edges[::2], edges[1::2], strict=True))
    merged: list[list[int]] = []
    gap = round(min_gap_s * sr / hop)
    for start, end in runs:
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    min_frames = max(1, round(min_dur_s * sr / hop))
    return [
        (start * hop / sr, min(end * hop, len(x)) / sr)
        for start, end in merged
        if end - start >= min_frames
    ]


# ---------------------------------------------------------------------------
# Time-base helpers
# ---------------------------------------------------------------------------


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling at the exact rate ratio where that is cheap.

    Standard audio rates reduce to small fractions (44.1k to 48k is 160/147), but an
    unusual pair can reduce to one with a huge denominator, which would make the
    polyphase filter enormous. Those fall back to an approximation.
    """
    if sr_in == sr_out:
        return x
    ratio = Fraction(int(sr_out), int(sr_in))
    if max(ratio.numerator, ratio.denominator) > 2000:
        ratio = ratio.limit_denominator(2000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator)


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


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def fade_edges(x: np.ndarray, sr: int, ms: float = 5.0) -> np.ndarray:
    n = max(1, int(sr * ms / 1000.0))
    if len(x) < 2 * n:
        return x
    ramp = np.hanning(2 * n)
    y = x.copy()
    y[:n] *= ramp[:n]
    y[-n:] *= ramp[n:]
    return y


def soft_clip(x: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Linear below ``knee``, then a tanh shoulder that never exceeds 1.0."""
    y = np.asarray(x, dtype=np.float64).copy()
    over = np.abs(y) > knee
    if np.any(over):
        span = 1.0 - knee
        mag = knee + span * np.tanh((np.abs(y[over]) - knee) / span)
        y[over] = np.sign(y[over]) * mag
    return y


# ---------------------------------------------------------------------------
# Sampling helpers for config ranges
# ---------------------------------------------------------------------------


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


def sample_int(rng: np.random.Generator, spec) -> int:
    if isinstance(spec, (int, float)):
        return int(spec)
    return int(rng.integers(int(spec[0]), int(spec[1]) + 1))


def choose(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Pick a key with probability proportional to its weight."""
    keys = [k for k, w in weights.items() if w > 0]
    if not keys:
        raise ValueError(f"no positive weights in {weights}")
    p = np.array([float(weights[k]) for k in keys])
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]
