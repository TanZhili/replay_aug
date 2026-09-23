"""Programme chains: what speech goes through before it reaches a loudspeaker.

broadcast   TV programmes. Mastering compression, loudness normalisation to EBU R128
            / ATSC A/85 (-23 / -24 LUFS), AAC / AC-3 / MP2 delivery, often a bed of
            music or effects under the dialogue.
streaming   Online video, short-video apps, podcasts. -12 to -18 LUFS, heavier
            compression, AAC / Opus / MP3.
phone_call  The far end of a call on speakerphone or car hands-free: telephone band,
            far-end AGC and background, AMR-NB / G.711 (narrowband) or G.722 / Opus
            (wideband).
radio       FM (pre-emphasis, dense multiband processing, clipping, 15 kHz low-pass,
            hiss) or digital radio (MP2 for DAB, AAC for DAB+).

Some programmes were recorded in a room of their own - a vlog, on-location drama, a
caller on speakerphone - so a share of them carries a small-room reverb before it
is played into the listener's room: two rooms stacked, which no live talker has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .codecs import CodecSpec, codec_roundtrip
from .dsp import (
    EPS,
    active_level_db,
    apply_sos,
    butter_sos,
    choose,
    rms,
    sample_range,
    soft_clip,
)
from .dynamics import compressor, limiter, multiband_compressor
from .irbank import synth_ir
from .loudness import normalize_loudness
from .noise import AudioBank, shaped_noise

CONTENT_PROFILES: dict[str, dict] = {
    "broadcast": {
        "loudness_lufs": [-24.0, -22.0],
        "compressor_prob": 0.7,
        "compressor_ratio": [1.5, 3.0],
        "compressor_threshold_rel_db": [-8.0, 0.0],
        "limiter_ceiling_db": [-2.0, -1.0],
        "codecs": {"aac": 0.4, "ac3": 0.2, "mp2": 0.2, "none": 0.2},
        "bitrate_kbps": {"aac": [64, 192], "ac3": [192, 384], "mp2": [128, 256]},
        "codec_rate": 48000,
        "bed_prob": 0.5,
        "bed_snr_db": [6.0, 20.0],
        "source_room_prob": 0.2,
    },
    "streaming": {
        "loudness_lufs": [-18.0, -12.0],
        "compressor_prob": 0.7,
        "compressor_ratio": [2.0, 5.0],
        "compressor_threshold_rel_db": [-10.0, -2.0],
        "limiter_ceiling_db": [-1.5, -0.3],
        "codecs": {"aac": 0.45, "opus": 0.4, "mp3": 0.15},
        "bitrate_kbps": {"aac": [48, 160], "opus": [24, 128], "mp3": [64, 192]},
        "codec_rate": 48000,
        "bed_prob": 0.4,
        "bed_snr_db": [4.0, 18.0],
        "source_room_prob": 0.4,
    },
    "phone_call": {
        "loudness_lufs": [-26.0, -18.0],
        "bands": {"narrow": 0.5, "wide": 0.5},
        "codecs_narrow": {"amr_nb": 0.6, "pcm_alaw": 0.2, "pcm_mulaw": 0.2},
        "codecs_wide": {"opus": 0.6, "g722": 0.4},
        "bitrate_kbps": {
            "amr_nb": [4.75, 12.2],
            "opus": [12, 32],
            "g722": [64, 64],
            "pcm_alaw": [64, 64],
            "pcm_mulaw": [64, 64],
        },
        "compressor_prob": 0.8,
        "compressor_ratio": [2.0, 4.0],
        "compressor_threshold_rel_db": [-10.0, -2.0],
        "limiter_ceiling_db": [-3.0, -1.0],
        "far_end_noise_prob": 0.5,
        "far_end_snr_db": [15.0, 35.0],
        "bed_prob": 0.0,
        "bed_snr_db": [10.0, 20.0],
        "source_room_prob": 0.5,
    },
    "radio": {
        "loudness_lufs": [-18.0, -12.0],
        "fm_prob": 0.6,
        "multiband_ratio": [2.5, 6.0],
        "multiband_threshold_rel_db": [-14.0, -6.0],
        "preemphasis_us": {"50": 0.7, "75": 0.3},
        "fm_hiss_snr_db": [35.0, 60.0],
        "codecs": {"mp2": 0.5, "aac": 0.5},
        "bitrate_kbps": {"mp2": [96, 192], "aac": [48, 96]},
        "codec_rate": 48000,
        "limiter_ceiling_db": [-1.0, -0.3],
        "bed_prob": 0.4,
        "bed_snr_db": [6.0, 18.0],
        "source_room_prob": 0.2,
    },
}


@dataclass
class ContentChain:
    profile: str
    params: dict = field(default_factory=dict)


def _pick_codec(rng, weights: dict[str, float], available: set[str]) -> str | None:
    usable = {k: w for k, w in weights.items() if k == "none" or k in available}
    if not usable:
        return None
    name = choose(rng, usable)
    return None if name == "none" else name


def sample_content(
    profile: str, preset: dict, rng: np.random.Generator, available_codecs: set[str], has_beds: bool
) -> ContentChain:
    """Draw concrete settings for one programme of the given kind."""
    p: dict = {"loudness_lufs": round(sample_range(rng, preset["loudness_lufs"]), 2)}
    if rng.random() < preset.get("source_room_prob", 0.0):
        p["source_room"] = {
            "t60_s": round(rng.uniform(0.15, 0.5), 3),
            "drr_db": round(rng.uniform(3.0, 15.0), 2),
        }
    if has_beds and rng.random() < preset.get("bed_prob", 0.0):
        p["bed_snr_db"] = round(sample_range(rng, preset["bed_snr_db"]), 2)

    if profile == "phone_call":
        band = choose(rng, preset["bands"])
        p["band"] = band
        codec = _pick_codec(rng, preset[f"codecs_{band}"], available_codecs)
        p["codec_rate"] = 8000 if band == "narrow" else 16000
        if rng.random() < preset["far_end_noise_prob"]:
            p["far_end_snr_db"] = round(sample_range(rng, preset["far_end_snr_db"]), 2)
    elif profile == "radio":
        p["fm"] = bool(rng.random() < preset["fm_prob"])
        p["multiband"] = {
            "ratio": round(sample_range(rng, preset["multiband_ratio"]), 2),
            "threshold_rel_db": round(sample_range(rng, preset["multiband_threshold_rel_db"]), 2),
        }
        if p["fm"]:
            codec = None
            p["preemphasis_us"] = float(choose(rng, preset["preemphasis_us"]))
            p["fm_hiss_snr_db"] = round(sample_range(rng, preset["fm_hiss_snr_db"]), 2)
        else:
            codec = _pick_codec(rng, preset["codecs"], available_codecs)
            p["codec_rate"] = preset.get("codec_rate", 48000)
    else:
        codec = _pick_codec(rng, preset["codecs"], available_codecs)
        p["codec_rate"] = preset.get("codec_rate", 48000)

    if codec is not None:
        lo, hi = preset["bitrate_kbps"].get(codec, [64, 64])
        p["codec"] = codec
        p["bitrate_kbps"] = round(float(rng.uniform(lo, hi)), 2)
    if profile != "radio" and rng.random() < preset.get("compressor_prob", 0.0):
        p["compressor"] = {
            "ratio": round(sample_range(rng, preset["compressor_ratio"]), 2),
            "threshold_rel_db": round(sample_range(rng, preset["compressor_threshold_rel_db"]), 2),
        }
    p["limiter_ceiling_db"] = round(sample_range(rng, preset["limiter_ceiling_db"]), 2)
    return ContentChain(profile, p)


def _emphasis_sos(sr: int, tau_us: float, inverse: bool = False) -> np.ndarray:
    """First-order FM pre-emphasis (or its de-emphasis inverse), made proper by a pole."""
    wz = 1.0 / (tau_us * 1e-6)
    wp = 2 * np.pi * min(0.45 * sr, 20000.0)
    z, p, k = [-wz], [-wp], wp / wz
    if inverse:
        z, p, k = p, z, 1.0 / k
    zd, pd, kd = signal.bilinear_zpk(z, p, k, sr)
    return signal.zpk2sos(zd, pd, kd)


def apply_content(
    x: np.ndarray,
    sr: int,
    chain: ContentChain,
    rng: np.random.Generator,
    beds: AudioBank | None = None,
) -> tuple[np.ndarray, dict]:
    """Run a programme through its chain. The result is in digital full scale."""
    p = chain.params
    y = np.asarray(x, dtype=np.float64)
    meta: dict = {
        "content": chain.profile,
        **{k: v for k, v in p.items() if not isinstance(v, dict)},
    }

    if "source_room" in p:
        room = p["source_room"]
        h = synth_ir(sr, room["t60_s"], room["drr_db"], rng)
        y = signal.fftconvolve(y, h)[: len(y)]
        meta["source_room"] = room

    speech_db = active_level_db(y, sr, weighted=False)
    if "far_end_snr_db" in p:
        hum = shaped_noise(len(y), sr, rng, lambda f: -3.0 * np.log2(np.maximum(f / 200.0, 1.0)))
        y = y + hum * 10.0 ** ((speech_db - p["far_end_snr_db"]) / 20.0)
    if "bed_snr_db" in p and beds:
        bed, label = beds.sample(len(y), rng)
        y = y + bed / (rms(bed) + EPS) * 10.0 ** ((speech_db - p["bed_snr_db"]) / 20.0)
        meta["bed"] = label

    if chain.profile == "phone_call":
        band = (300.0, 3400.0) if p["band"] == "narrow" else (80.0, 7000.0)
        y = apply_sos(y, butter_sos(sr, band, "band", 4))

    if "compressor" in p:
        comp = p["compressor"]
        threshold = active_level_db(y, sr, weighted=False) + comp["threshold_rel_db"]
        y, _ = compressor(y, sr, threshold, comp["ratio"], attack_ms=10.0, release_ms=200.0)
        meta["compressor"] = comp

    if chain.profile == "radio":
        mb = p["multiband"]
        threshold = active_level_db(y, sr, weighted=False) + mb["threshold_rel_db"]
        y, _ = multiband_compressor(y, sr, (200.0, 2000.0), threshold, mb["ratio"])
        meta["multiband"] = mb

    y, _ = normalize_loudness(y, sr, p["loudness_lufs"])

    if chain.profile == "radio" and p.get("fm"):
        y = apply_sos(y, _emphasis_sos(sr, p["preemphasis_us"]))
        y = soft_clip(y / 0.9, knee=0.7) * 0.9
        y = apply_sos(y, butter_sos(sr, min(15000.0, 0.45 * sr), "low", 8))
        y = apply_sos(y, _emphasis_sos(sr, p["preemphasis_us"], inverse=True))
        hiss = apply_sos(
            np.asarray(rng.standard_normal(len(y))),
            butter_sos(sr, min(15000.0, 0.45 * sr), "low", 4),
        )
        y = y + hiss / (rms(hiss) + EPS) * 10.0 ** (
            (active_level_db(y, sr, weighted=False) - p["fm_hiss_snr_db"]) / 20.0
        )

    y, _ = limiter(y, sr, p["limiter_ceiling_db"], lookahead_ms=1.5, release_ms=50.0)

    if "codec" in p:
        spec = CodecSpec(p["codec"], int(p["bitrate_kbps"] * 1000), p.get("codec_rate"))
        y, codec_meta = codec_roundtrip(y, sr, spec)
        meta.update(codec_meta)
    return y, meta


__all__ = ["CONTENT_PROFILES", "ContentChain", "apply_content", "sample_content"]
