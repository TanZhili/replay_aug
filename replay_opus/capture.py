"""Capture side: the microphone a recording passes through.

A scene's capture is the device your full-duplex model listens through. Everything
in it is shared by every source in the scene, which is the point: a live talker and
a TV reach the model through the same MEMS microphone, the same self-noise, the same
gain staging, the same front-end high-pass, noise suppression and AGC, and the same
converter. Whatever differs between the classes must come from the sources.

The same model describes the other microphones a voice can meet before a loudspeaker
plays it: the phone that recorded a short video in someone else's room, or the phone
that recorded a person in this very room. Those get capture chains of their own
(``playback.rerecorded.capture`` / ``playback.replay.capture``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .codecs import CodecSpec, codec_roundtrip
from .denoise import suppress_noise
from .dsp import (
    P_REF,
    apply_sos,
    butter_sos,
    cascade,
    choose,
    peaking_sos,
    resample_to,
    sample_range,
)
from .dynamics import agc
from .noise import scale_to_dba


@dataclass
class Capture:
    fullscale_spl_db: float  # SPL of a full-scale sine at the converter, after all gain
    self_noise_dba: float
    mic_sos: np.ndarray
    mic_ir: np.ndarray | None
    highpass_hz: float | None
    agc: dict | None
    uplink: dict | None
    ns: dict | None = None
    meta: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "fullscale_spl_db": round(self.fullscale_spl_db, 2),
            "self_noise_dba": round(self.self_noise_dba, 2),
            "highpass_hz": None if self.highpass_hz is None else round(self.highpass_hz, 1),
            "ns": self.ns,
            "agc": self.agc,
            "uplink": self.uplink,
            **self.meta,
        }


def mic_response_sos(sr: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A MEMS capsule: low-frequency vent roll-off, port resonance, small tilt."""
    hp = float(np.exp(rng.uniform(np.log(30.0), np.log(150.0))))
    order = int(rng.choice([1, 2]))
    res_f = rng.uniform(9000.0, 15000.0)
    res_db = rng.uniform(1.0, 6.0)
    parts = [
        butter_sos(sr, hp, "high", order),
        peaking_sos(sr, res_f, res_db, rng.uniform(1.5, 4.0)),
    ]
    tilt = []
    for _ in range(2):
        f0 = float(np.exp(rng.uniform(np.log(200.0), np.log(6000.0))))
        g = rng.uniform(-1.5, 1.5)
        parts.append(peaking_sos(sr, f0, g, rng.uniform(0.5, 1.5)))
        tilt.append({"f0_hz": round(f0, 1), "gain_db": round(g, 2)})
    return cascade(*parts), {
        "mic_hp_hz": round(hp, 1),
        "mic_hp_order": order,
        "mic_resonance_hz": round(res_f, 1),
        "mic_resonance_db": round(res_db, 2),
        "mic_tilt": tilt,
    }


def sample_capture(
    cfg, sr: int, rng: np.random.Generator, mic_irs=None, available_codecs=()
) -> Capture:
    sos, meta = mic_response_sos(sr, rng)
    mic_ir = None
    if mic_irs is not None and len(mic_irs):
        mic_ir, label = mic_irs.sample(rng)
        sos = np.zeros((0, 6))
        meta = {"mic_ir": label}
    highpass = sample_range(rng, cfg.highpass_hz) if rng.random() < cfg.highpass_prob else None
    agc_cfg = None
    if rng.random() < cfg.agc_prob:
        agc_cfg = {
            "target_dbfs": round(sample_range(rng, cfg.agc_target_dbfs), 2),
            "max_gain_db": float(cfg.agc_max_gain_db),
        }
    uplink = None
    usable = {k: w for k, w in cfg.uplink_codecs.items() if k in available_codecs}
    if usable and rng.random() < cfg.uplink_prob:
        uplink = {
            "codec": choose(rng, usable),
            "bitrate_kbps": round(sample_range(rng, cfg.uplink_bitrate_kbps), 2),
        }
    fullscale = sample_range(rng, cfg.fullscale_spl_db)
    self_noise = sample_range(rng, cfg.self_noise_dba)
    ns = None
    if rng.random() < cfg.ns_prob:
        ns = {"max_db": round(sample_range(rng, cfg.ns_max_db), 2)}
    return Capture(
        fullscale_spl_db=fullscale,
        self_noise_dba=self_noise,
        mic_sos=sos,
        mic_ir=mic_ir,
        highpass_hz=highpass,
        agc=agc_cfg,
        uplink=uplink,
        ns=ns,
        meta=meta,
    )


def to_digital(p_mic: np.ndarray, sr: int, cap: Capture) -> np.ndarray:
    """Microphone response and gain staging: pascals in, full-scale units out."""
    x = np.asarray(p_mic, dtype=np.float64)
    if cap.mic_ir is not None:
        x = signal.fftconvolve(x, cap.mic_ir)[: len(x)]
    x = apply_sos(x, cap.mic_sos)
    return x / (np.sqrt(2.0) * P_REF * 10.0 ** (cap.fullscale_spl_db / 20.0))


def _uplink(x: np.ndarray, sr: int, cap: Capture, meta: dict) -> np.ndarray:
    if cap.uplink is None:
        return x
    spec = CodecSpec(cap.uplink["codec"], int(cap.uplink["bitrate_kbps"] * 1000))
    x, codec_meta = codec_roundtrip(x, sr, spec)
    meta["uplink_codec_sr"] = codec_meta["codec_sr"]
    return x


def capture(
    p_mic: np.ndarray, sr: int, out_sr: int, cap: Capture, noise_rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Self-noise, microphone, gain, high-pass, noise suppression, AGC, converter, uplink."""
    self_noise = scale_to_dba(noise_rng.standard_normal(len(p_mic)), sr, cap.self_noise_dba)
    x = to_digital(p_mic + self_noise, sr, cap)
    meta: dict = {}
    if cap.highpass_hz is not None:
        x = apply_sos(x, butter_sos(sr, cap.highpass_hz, "high", 2))
    if cap.ns is not None:
        x, ns_meta = suppress_noise(x, sr, cap.ns["max_db"])
        meta["ns_applied"] = ns_meta["ns_applied"]
    if cap.agc is not None:
        x, agc_meta = agc(x, sr, cap.agc["target_dbfs"], cap.agc["max_gain_db"])
        meta.update(agc_meta)
    clipped = int(np.sum(np.abs(x) > 1.0))
    x = np.clip(x, -1.0, 1.0)
    meta["clipped_pct"] = round(100.0 * clipped / max(len(x), 1), 4)
    x = _uplink(x, sr, cap, meta)
    y = resample_to(x, sr, out_sr)
    return np.clip(y, -1.0, 1.0), meta


def digital_front_end(x: np.ndarray, sr: int, out_sr: int, cap: Capture) -> tuple[np.ndarray, dict]:
    """Only the digital stages of a capture chain - high-pass and uplink codec - for
    audio that a real microphone has already recorded."""
    y = np.asarray(x, dtype=np.float64)
    meta: dict = {}
    if cap.highpass_hz is not None:
        y = apply_sos(y, butter_sos(sr, cap.highpass_hz, "high", 2))
    y = _uplink(np.clip(y, -1.0, 1.0), sr, cap, meta)
    return resample_to(y, sr, out_sr), meta


__all__ = [
    "Capture",
    "capture",
    "digital_front_end",
    "mic_response_sos",
    "sample_capture",
    "to_digital",
]
