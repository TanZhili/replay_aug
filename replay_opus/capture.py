"""Capture stage: microphone response, converter clipping, recorder-side gain control.

This is the part that pure RIR convolution skips, and it is a large part of why
RIR-only augmentation recovers only some of the replay effect. A phone or a meeting
client runs high-pass -> AEC -> noise suppression -> AGC before anything reaches disk.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .dsp import EPS, butter, random_spectral_tilt, set_peak_db


def apply_mic_response(
    x: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    mic_ir: np.ndarray | None = None,
    hp_hz: tuple[float, float] = (40.0, 140.0),
    lp_hz: tuple[float, float] = (7000.0, 18000.0),
    tilt_bands: int = 3,
    tilt_db: float = 4.0,
) -> tuple[np.ndarray, dict]:
    """Convolve a measured microphone IR, or synthesise a plausible one.

    Measured IRs for 20 phone models are available from POLIPHONE; pass one in via
    ``mic_ir`` to make the output sound like that specific handset.
    """
    if mic_ir is not None:
        return signal.fftconvolve(x, mic_ir, mode="full"), {"mic": "measured_ir"}

    hp = float(rng.uniform(*hp_hz))
    lp = float(rng.uniform(*lp_hz))
    y = butter(x, sr, hp, kind="high", order=2)
    y = butter(y, sr, min(lp, 0.98 * sr / 2), kind="low", order=4)
    y, tilt = random_spectral_tilt(y, sr, rng, n_bands=tilt_bands, max_gain_db=tilt_db)
    return y, {
        "mic": "synthetic",
        "mic_hp_hz": round(hp, 1),
        "mic_lp_hz": round(lp, 1),
        "mic_tilt": tilt,
    }


def simple_agc(
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

    A simplification of WebRTC AGC2; for bit-accurate behaviour run the real thing
    through the ``pywebrtc-audio`` bindings.
    """
    hop = max(1, int(sr * frame_ms / 1000.0))
    target_rms = 10.0 ** (target_dbfs / 20.0)
    max_gain = 10.0 ** (max_gain_db / 20.0)
    decay = 10.0 ** (-level_decay_db_s * frame_ms / 1000.0 / 20.0)
    a_attack = np.exp(-frame_ms / max(attack_ms, 1e-3))
    a_release = np.exp(-frame_ms / max(release_ms, 1e-3))

    n_frames = int(np.ceil(len(x) / hop))
    padded = np.pad(x, (0, n_frames * hop - len(x)))
    frame_level = np.sqrt(np.mean(padded.reshape(n_frames, hop) ** 2, axis=1) + EPS)
    # Never chase a level more than 40 dB below the loud passages: that region is
    # noise, and tracking it is what produces the pumping described above.
    floor = float(np.percentile(frame_level, 90)) * 10.0 ** (-40.0 / 20.0)

    gains = np.ones(n_frames)
    speech_level = max(float(frame_level[0]), floor)
    gain = 1.0
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


def adc_clip(x: np.ndarray, headroom_db: float = -1.0) -> tuple[np.ndarray, dict]:
    """Set the recording level relative to full scale, then hard-clip.

    ``headroom_db`` is where the signal peak lands: negative leaves headroom, positive
    means the level was set too hot and the peaks clip. Real recordings are levelled
    by a person or an AGC rather than by the signal, so a fraction of them do clip -
    a config range spanning zero reproduces that.
    """
    y = set_peak_db(x, headroom_db)
    clipped = int(np.sum(np.abs(y) > 1.0))
    y = np.clip(y, -1.0, 1.0)
    return y, {
        "adc_headroom_db": round(float(headroom_db), 2),
        "clipped_samples": clipped,
        "clipped_pct": round(100.0 * clipped / max(len(y), 1), 4),
    }
