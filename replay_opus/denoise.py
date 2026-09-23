"""Noise suppression of the kind phones and call front ends run.

Calls, voice messages and many recorder apps pass the microphone signal through a
suppressor before it is stored or sent: stationary noise drops between words, and a
little of the room's reverberant tail goes with it. A voice recorded that way and
played back later carries those marks; a device whose own front end suppresses noise
puts them on everything it hears.

A short-time spectral gain. The noise spectrum is estimated per bin from the quietest
frames of the whole signal (this is offline data generation, not a causal device),
the a-priori SNR follows the decision-directed rule of Ephraim & Malah (1984), and
the gain is a Wiener gain with a floor, so suppression never exceeds ``max_db``.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .dsp import EPS


def suppress_noise(
    x: np.ndarray,
    sr: int,
    max_db: float = 12.0,
    frame_ms: float = 20.0,
    noise_percentile: float = 15.0,
    smoothing: float = 0.96,
) -> tuple[np.ndarray, dict]:
    """Suppress stationary noise by at most ``max_db``. Returns ``(y, meta)``."""
    x = np.asarray(x, dtype=np.float64)
    nperseg = int(2 ** np.round(np.log2(max(sr * frame_ms / 1000.0, 64.0))))
    if len(x) < 4 * nperseg:
        return x.copy(), {"ns_max_db": round(float(max_db), 2), "ns_applied": False}
    hop = nperseg // 2
    _, _, spec = signal.stft(x, fs=sr, window="hann", nperseg=nperseg, noverlap=hop)
    power = np.abs(spec) ** 2
    # A noise bin's power is exponentially distributed, so a low percentile sits well
    # below the mean: divide by that percentile of the unit exponential.
    q = noise_percentile / 100.0
    noise = np.percentile(power, noise_percentile, axis=1) / -np.log1p(-q) + EPS
    posterior = power / noise[:, None]
    floor = 10.0 ** (-max_db / 20.0)
    gains = np.empty_like(power)
    previous = np.zeros(power.shape[0])
    for t in range(power.shape[1]):
        prior = smoothing * previous + (1.0 - smoothing) * np.maximum(posterior[:, t] - 1.0, 0.0)
        gain = np.maximum(prior / (1.0 + prior), floor)
        gains[:, t] = gain
        previous = gain**2 * posterior[:, t]
    _, y = signal.istft(spec * gains, fs=sr, window="hann", nperseg=nperseg, noverlap=hop)
    y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    return y, {"ns_max_db": round(float(max_db), 2), "ns_applied": True}


__all__ = ["suppress_noise"]
