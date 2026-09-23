"""Acoustic descriptors used to control IR synthesis and to validate the output.

The T60 / DRR / C50 definitions follow the conventions used by the ASVspoof PA
and ACE Challenge corpora, so numbers here are comparable with their metadata.
"""

from __future__ import annotations

import numpy as np

from .dsp import EPS

DIRECT_WINDOW_MS = 2.5


def direct_index(h: np.ndarray) -> int:
    return int(np.argmax(np.abs(h)))


def energy_decay_curve(h: np.ndarray) -> np.ndarray:
    """Schroeder backward-integrated decay curve, normalized to 0 dB."""
    energy = np.square(h)
    edc = np.cumsum(energy[::-1])[::-1]
    edc = edc / (edc[0] + EPS)
    return 10.0 * np.log10(edc + EPS)


def estimate_t60(h: np.ndarray, sr: int, upper_db: float = -5.0, lower_db: float = -35.0) -> float:
    """T60 extrapolated from the -5..-35 dB slope of the decay curve (i.e. T30)."""
    h = np.asarray(h, dtype=np.float64)
    d0 = direct_index(h)
    edc = energy_decay_curve(h[d0:])
    try:
        i_hi = int(np.argmax(edc <= upper_db))
        i_lo = int(np.argmax(edc <= lower_db))
    except ValueError:
        return float("nan")
    if i_lo <= i_hi + 1:
        return float("nan")
    t = np.arange(i_hi, i_lo) / sr
    slope, _ = np.polyfit(t, edc[i_hi:i_lo], 1)
    if slope >= -EPS:
        return float("nan")
    return float(-60.0 / slope)


def split_direct_late(h: np.ndarray, sr: int, window_ms: float = DIRECT_WINDOW_MS):
    """Split into direct (``t_d +/- window``) and late components.

    This is the decomposition Bryan (ICASSP 2020) uses for IR augmentation, and
    the same one that defines DRR.
    """
    h = np.asarray(h, dtype=np.float64)
    d0 = direct_index(h)
    half = max(1, int(sr * window_ms / 1000.0))
    lo, hi = max(0, d0 - half), min(len(h), d0 + half + 1)
    direct = np.zeros_like(h)
    late = np.zeros_like(h)
    direct[lo:hi] = h[lo:hi]
    late[hi:] = h[hi:]
    return direct, late, d0, hi


def estimate_drr(h: np.ndarray, sr: int, window_ms: float = DIRECT_WINDOW_MS) -> float:
    direct, late, _, _ = split_direct_late(h, sr, window_ms)
    return float(10.0 * np.log10((np.sum(direct**2) + EPS) / (np.sum(late**2) + EPS)))


def estimate_c50(h: np.ndarray, sr: int, early_ms: float = 50.0) -> float:
    h = np.asarray(h, dtype=np.float64)
    d0 = direct_index(h)
    split = d0 + int(sr * early_ms / 1000.0)
    early = np.sum(h[d0:split] ** 2)
    late = np.sum(h[split:] ** 2)
    return float(10.0 * np.log10((early + EPS) / (late + EPS)))


def describe_ir(h: np.ndarray, sr: int) -> dict:
    return {
        "t60_s": round(estimate_t60(h, sr), 4),
        "drr_db": round(estimate_drr(h, sr), 2),
        "c50_db": round(estimate_c50(h, sr), 2),
        "length_ms": round(1000.0 * len(h) / sr, 1),
    }
