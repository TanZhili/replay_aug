"""Programme loudness per ITU-R BS.1770-4 (mono), as used by EBU R128 and ATSC A/85.

Broadcasters normalise to -23 LUFS (EBU R128) or -24 LKFS (ATSC A/85); streaming
platforms land around -14 to -16 LUFS. Content played by a TV or a phone has been
through one of these targets, which sets how hard the playback device is driven.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal

from .dsp import EPS

ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0


@lru_cache(maxsize=16)
def k_weighting_sos(sr: int) -> np.ndarray:
    """The BS.1770 pre-filter (high shelf) and RLB high-pass, designed for any rate.

    Same parametrisation as libebur128, which reproduces the standard's published
    48 kHz coefficients exactly and extends them to other rates.
    """
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = np.array(
        [
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
            1.0,
            2.0 * (k * k - 1.0) / a0,
            (1.0 - k / q + k * k) / a0,
        ]
    )

    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a0 = 1.0 + k / q + k * k
    highpass = np.array([1.0, -2.0, 1.0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    return np.vstack([shelf, highpass])


def integrated_loudness(x: np.ndarray, sr: int) -> float:
    """Gated integrated loudness in LUFS. Returns -inf for silence."""
    y = signal.sosfilt(k_weighting_sos(sr), np.asarray(x, dtype=np.float64))
    block = round(0.4 * sr)
    step = round(0.1 * sr)
    if len(y) < block:
        powers = np.array([np.mean(y**2)])
    else:
        starts = np.arange(0, len(y) - block + 1, step)
        csum = np.concatenate([[0.0], np.cumsum(y**2)])
        powers = (csum[starts + block] - csum[starts]) / block

    loud = -0.691 + 10.0 * np.log10(powers + EPS)
    gated = powers[loud > ABSOLUTE_GATE_LUFS]
    if gated.size == 0:
        return float("-inf")
    relative = -0.691 + 10.0 * np.log10(np.mean(gated) + EPS) + RELATIVE_GATE_LU
    keep = powers[(loud > ABSOLUTE_GATE_LUFS) & (loud > relative)]
    if keep.size == 0:
        return float("-inf")
    return float(-0.691 + 10.0 * np.log10(np.mean(keep) + EPS))


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float) -> tuple[np.ndarray, float]:
    """Scale to a target loudness. Returns ``(y, gain_db)``."""
    current = integrated_loudness(x, sr)
    if not np.isfinite(current):
        return x, 0.0
    gain_db = target_lufs - current
    return x * 10.0 ** (gain_db / 20.0), float(gain_db)
