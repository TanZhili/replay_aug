"""Dynamics processors: compressor, look-ahead limiter, multiband compressor, AGC.

Programme material, playback devices and capture devices all run some of these:
broadcast and streaming mastering compress and limit, a TV's auto-volume and a
phone's smart amplifier compress again, and the capturing device's AGC levels
whatever reaches the microphone. Gain is computed on short frames and applied per
sample, which keeps the attack/release recursion cheap in pure numpy.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .dsp import EPS, apply_sos, linkwitz_riley_sos


def _frames(x: np.ndarray, hop: int) -> tuple[np.ndarray, int]:
    n_frames = int(np.ceil(len(x) / hop))
    padded = np.pad(x, (0, n_frames * hop - len(x)))
    return padded.reshape(n_frames, hop), n_frames


def _knee_curve(
    level_db: np.ndarray, threshold_db: float, ratio: float, knee_db: float
) -> np.ndarray:
    """Static gain (dB, <= 0) of a soft-knee compressor."""
    over = level_db - threshold_db
    slope = 1.0 / max(ratio, 1.0) - 1.0
    gain = np.zeros_like(level_db)
    if knee_db > 0:
        in_knee = np.abs(over) <= knee_db / 2
        gain[in_knee] = slope * (over[in_knee] + knee_db / 2) ** 2 / (2 * knee_db)
        above = over > knee_db / 2
    else:
        above = over > 0
    gain[above] = slope * over[above]
    return gain


def _smooth_gain(target_db: np.ndarray, a_attack: float, a_release: float) -> np.ndarray:
    out = np.empty_like(target_db)
    g = 0.0
    for i, t in enumerate(target_db):
        a = a_attack if t < g else a_release
        g = a * g + (1.0 - a) * t
        out[i] = g
    return out


def compressor(
    x: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float = 10.0,
    release_ms: float = 150.0,
    knee_db: float = 6.0,
    makeup_db: float = 0.0,
    hop_ms: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Feed-forward RMS compressor. ``threshold_db`` is in dBFS (RMS of a frame)."""
    hop = max(1, int(sr * hop_ms / 1000.0))
    frames, n_frames = _frames(x, hop)
    level_db = 10.0 * np.log10(np.mean(frames**2, axis=1) + EPS)
    target = _knee_curve(level_db, threshold_db, ratio, knee_db)
    gain_db = _smooth_gain(
        target, np.exp(-hop_ms / max(attack_ms, 1e-3)), np.exp(-hop_ms / max(release_ms, 1e-3))
    )
    centres = (np.arange(n_frames) + 0.5) * hop
    envelope = np.interp(np.arange(len(x)), centres, 10.0 ** ((gain_db + makeup_db) / 20.0))
    return x * envelope, {
        "threshold_db": round(float(threshold_db), 2),
        "ratio": round(float(ratio), 2),
        "max_reduction_db": round(float(-gain_db.min()), 2),
    }


def limiter(
    x: np.ndarray,
    sr: int,
    ceiling_db: float = -1.0,
    lookahead_ms: float = 3.0,
    release_ms: float = 80.0,
    hop_ms: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Look-ahead peak limiter; the output never exceeds ``ceiling_db``.

    The gain drops ahead of each peak (a running minimum over the look-ahead window,
    then a box filter of the same length), so the peak is caught without a hard
    gain step, and recovers with the release time constant.
    """
    ceiling = 10.0 ** (ceiling_db / 20.0)
    hop = max(1, int(sr * hop_ms / 1000.0))
    frames, n_frames = _frames(x, hop)
    peaks = np.max(np.abs(frames), axis=1)
    need = np.minimum(1.0, ceiling / (peaks + EPS))

    look = max(1, round(lookahead_ms / hop_ms))
    ahead = sliding_window_view(np.concatenate([need, np.ones(look)]), look + 1).min(axis=1)

    a_release = np.exp(-hop_ms / max(release_ms, 1e-3))
    held = np.empty(n_frames)
    g = 1.0
    for i, r in enumerate(ahead):
        g = r if r < g else a_release * g + (1.0 - a_release) * r
        held[i] = g
    # Causal moving average over look+1 frames, primed with unity gain: at a peak
    # frame every averaged value already includes that peak's requirement.
    csum = np.concatenate([[0.0], np.cumsum(np.concatenate([np.ones(look), held]))])
    smoothed = np.minimum((csum[look + 1 :] - csum[: -look - 1]) / (look + 1), 1.0)

    y = x * np.repeat(smoothed, hop)[: len(x)]
    clipped = int(np.sum(np.abs(y) > ceiling))
    y = np.clip(y, -ceiling, ceiling)
    return y, {
        "ceiling_db": round(float(ceiling_db), 2),
        "max_reduction_db": round(float(-20.0 * np.log10(smoothed.min() + EPS)), 2),
        "residual_clipped": clipped,
    }


def multiband_compressor(
    x: np.ndarray,
    sr: int,
    crossovers_hz: tuple[float, ...],
    threshold_db: float,
    ratio: float,
    attack_ms: float = 5.0,
    release_ms: float = 120.0,
) -> tuple[np.ndarray, dict]:
    """Split by complementary subtraction (bands sum back exactly), compress each.

    This is the structure of broadcast processors such as the ones FM stations run,
    which is why radio speech sounds dense even between words.
    """
    bands = []
    rest = np.asarray(x, dtype=np.float64)
    for fc in crossovers_hz:
        low = apply_sos(rest, linkwitz_riley_sos(sr, fc, "low"))
        bands.append(low)
        rest = rest - low
    bands.append(rest)

    out = np.zeros_like(rest)
    reductions = []
    for band in bands:
        level = 10.0 * np.log10(np.mean(band**2) + EPS)
        compressed, meta = compressor(
            band, sr, min(threshold_db, level + 3.0), ratio, attack_ms, release_ms, knee_db=4.0
        )
        out += compressed
        reductions.append(meta["max_reduction_db"])
    return out, {
        "crossovers_hz": [round(float(f), 1) for f in crossovers_hz],
        "ratio": round(float(ratio), 2),
        "band_max_reduction_db": reductions,
    }


def agc(
    x: np.ndarray,
    sr: int,
    target_dbfs: float = -20.0,
    max_gain_db: float = 30.0,
    attack_ms: float = 20.0,
    release_ms: float = 400.0,
    frame_ms: float = 10.0,
    level_decay_db_s: float = 6.0,
) -> tuple[np.ndarray, dict]:
    """Automatic gain control driven by a decaying estimate of the speech level.

    The gain follows a peak-hold estimate of the *speech* level, not the level of the
    current frame. Driving it from frame level directly makes the AGC spend its whole
    range on pauses and lift the noise floor above the speech - pumping that no real
    recorder produces. Here the estimate falls at ``level_decay_db_s``, so a pause
    breathes a few dB rather than tens.

    A simplification of WebRTC AGC2.
    """
    hop = max(1, int(sr * frame_ms / 1000.0))
    target_rms = 10.0 ** (target_dbfs / 20.0)
    max_gain = 10.0 ** (max_gain_db / 20.0)
    decay = 10.0 ** (-level_decay_db_s * frame_ms / 1000.0 / 20.0)
    a_attack = np.exp(-frame_ms / max(attack_ms, 1e-3))
    a_release = np.exp(-frame_ms / max(release_ms, 1e-3))

    frames, n_frames = _frames(x, hop)
    frame_level = np.sqrt(np.mean(frames**2, axis=1) + EPS)
    typical = float(np.percentile(frame_level, 90))
    # Never chase a level more than 40 dB below the loud passages: that region is
    # noise, and tracking it is what produces the pumping described above.
    floor = typical * 10.0 ** (-40.0 / 20.0)

    # Warm start: a device's AGC has been running long before any clip begins, so it
    # starts settled on the typical level instead of ramping up from unity gain.
    gains = np.ones(n_frames)
    speech_level = max(typical, floor)
    gain = float(np.clip(target_rms / speech_level, 1.0 / max_gain, max_gain))
    for i in range(n_frames):
        speech_level = max(float(frame_level[i]), speech_level * decay, floor)
        wanted = float(np.clip(target_rms / speech_level, 1.0 / max_gain, max_gain))
        alpha = a_attack if wanted < gain else a_release
        gain = alpha * gain + (1.0 - alpha) * wanted
        gains[i] = gain

    envelope = np.repeat(gains, hop)[: len(x)]
    return x * envelope, {
        "agc_target_dbfs": round(float(target_dbfs), 2),
        "agc_gain_db_range": [
            round(float(20.0 * np.log10(gains.min() + EPS)), 2),
            round(float(20.0 * np.log10(gains.max() + EPS)), 2),
        ],
    }
