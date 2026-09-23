"""Playback devices: what a TV, a phone or a car audio system does to speech.

A device instance is sampled from its type preset and applied as

    programme -> device DSP (dialogue EQ, virtual bass, auto-volume compressor,
    smart-amp limiter) -> driver non-linearity -> per-driver response (crossover,
    high-pass, resonance, cavity peak, cabinet ripple, roll-off)

giving one pressure signal per driver, at 1 m on-axis. The scene then puts each
driver at its own position with its own directivity, so a stereo TV or a door/dash
car system reaches the microphone as several spatially separate copies.

Distortion is tied to how hard the device is driven: the programme is placed at
``level_dba - max_spl_db`` below full scale, so a phone playing loud video sits in its
limiter and non-linearity while a soundbar at the same level stays clean. The level
at 1 m is then calibrated exactly, so distortion and loudness are set independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .acoustics import N_BANDS, Directivity
from .dsp import (
    EPS,
    P_REF,
    a_weight,
    active_level_db,
    apply_sos,
    butter_sos,
    cascade,
    frame_energy,
    linkwitz_riley_sos,
    peaking_sos,
    resample_to,
    rms,
    sample_int,
    sample_range,
    soft_clip,
    speech_level_db,
    speech_level_from_energy,
    to_mono,
)
from .dynamics import compressor, limiter
from .environment import Placement

FULL_SCALE_SINE_DB = -3.0103  # RMS of a full-scale sine, dBFS

_SMALL_DRIVER = [1.0, 1.0, 0.98, 0.92, 0.82, 0.7, 0.6]
_CABINET = [1.0, 0.97, 0.92, 0.84, 0.74, 0.64, 0.56]

DEVICE_PRESETS: dict[str, dict] = {
    # Flat panels: small down- or rear-firing drivers in a slot, often DSP-heavy.
    "tv": {
        "hp_hz": [90.0, 220.0],
        "hp_order": [2, 4],
        "resonance_db": [1.0, 5.0],
        "lp_hz": [9000.0, 16000.0],
        "lp_order": [2, 4],
        "ripple_db": 4.0,
        "ripples": [3, 5],
        "cavity_prob": 0.3,
        "cavity_hz": [1800.0, 5000.0],
        "cavity_db": [2.0, 6.0],
        "max_spl_db": [82.0, 92.0],
        "level_dba": [55.0, 70.0],
        "thd_full_scale": [0.01, 0.04],
        "virtual_bass_prob": 0.4,
        "dialog_eq_prob": 0.5,
        "drc_prob": 0.5,
        "limiter_prob": 0.9,
        "directivity": [1.0, 1.0, 0.95, 0.85, 0.72, 0.6, 0.55],
        "content": {"broadcast": 0.6, "streaming": 0.4},
    },
    "soundbar": {
        "hp_hz": [50.0, 110.0],
        "hp_order": [2, 4],
        "resonance_db": [0.5, 3.0],
        "lp_hz": [14000.0, 20000.0],
        "lp_order": [2, 2],
        "ripple_db": 2.5,
        "ripples": [2, 4],
        "cavity_prob": 0.1,
        "cavity_hz": [2000.0, 5000.0],
        "cavity_db": [1.0, 3.0],
        "max_spl_db": [90.0, 100.0],
        "level_dba": [56.0, 72.0],
        "thd_full_scale": [0.003, 0.015],
        "virtual_bass_prob": 0.15,
        "dialog_eq_prob": 0.6,
        "drc_prob": 0.4,
        "limiter_prob": 0.8,
        "directivity": _CABINET,
        "content": {"broadcast": 0.5, "streaming": 0.5},
    },
    # Micro-speakers: a strongly damped resonance at 600-1000 Hz, steep roll-off
    # below it, a front-cavity peak a few kHz up, smart-amp protection on top.
    "phone": {
        "hp_hz": [400.0, 900.0],
        "hp_order": [2, 4],
        "resonance_db": [3.0, 9.0],
        "lp_hz": [8000.0, 14000.0],
        "lp_order": [2, 4],
        "ripple_db": 5.0,
        "ripples": [2, 4],
        "cavity_prob": 0.8,
        "cavity_hz": [2500.0, 6000.0],
        "cavity_db": [3.0, 8.0],
        "max_spl_db": [66.0, 76.0],
        "level_dba": [50.0, 66.0],
        "thd_full_scale": [0.03, 0.10],
        "virtual_bass_prob": 0.5,
        "dialog_eq_prob": 0.2,
        "drc_prob": 0.6,
        "limiter_prob": 1.0,
        "directivity": _SMALL_DRIVER,
        "content": {"streaming": 0.6, "phone_call": 0.4},
    },
    "tablet": {
        "hp_hz": [250.0, 550.0],
        "hp_order": [2, 4],
        "resonance_db": [2.0, 7.0],
        "lp_hz": [9000.0, 15000.0],
        "lp_order": [2, 4],
        "ripple_db": 4.5,
        "ripples": [2, 4],
        "cavity_prob": 0.6,
        "cavity_hz": [2500.0, 6000.0],
        "cavity_db": [2.0, 6.0],
        "max_spl_db": [72.0, 82.0],
        "level_dba": [52.0, 68.0],
        "thd_full_scale": [0.02, 0.06],
        "virtual_bass_prob": 0.5,
        "dialog_eq_prob": 0.2,
        "drc_prob": 0.5,
        "limiter_prob": 1.0,
        "directivity": _SMALL_DRIVER,
        "content": {"streaming": 0.85, "phone_call": 0.15},
    },
    "laptop": {
        "hp_hz": [180.0, 450.0],
        "hp_order": [2, 4],
        "resonance_db": [2.0, 6.0],
        "lp_hz": [9000.0, 16000.0],
        "lp_order": [2, 4],
        "ripple_db": 5.0,
        "ripples": [3, 5],
        "cavity_prob": 0.4,
        "cavity_hz": [2000.0, 5000.0],
        "cavity_db": [2.0, 6.0],
        "max_spl_db": [74.0, 84.0],
        "level_dba": [52.0, 68.0],
        "thd_full_scale": [0.02, 0.06],
        "virtual_bass_prob": 0.5,
        "dialog_eq_prob": 0.3,
        "drc_prob": 0.5,
        "limiter_prob": 1.0,
        "directivity": _SMALL_DRIVER,
        "content": {"streaming": 0.8, "phone_call": 0.2},
    },
    "smart_speaker": {
        "hp_hz": [55.0, 160.0],
        "hp_order": [2, 4],
        "resonance_db": [1.0, 4.0],
        "lp_hz": [10000.0, 18000.0],
        "lp_order": [2, 4],
        "ripple_db": 3.0,
        "ripples": [2, 4],
        "cavity_prob": 0.2,
        "cavity_hz": [2000.0, 5000.0],
        "cavity_db": [1.0, 4.0],
        "max_spl_db": [82.0, 95.0],
        "level_dba": [54.0, 70.0],
        "thd_full_scale": [0.005, 0.03],
        "virtual_bass_prob": 0.3,
        "dialog_eq_prob": 0.2,
        "drc_prob": 0.5,
        "limiter_prob": 1.0,
        "directivity": [1.0, 1.0, 1.0, 0.98, 0.94, 0.88, 0.8],
        "content": {"streaming": 0.8, "phone_call": 0.2},
    },
    "bluetooth_speaker": {
        "hp_hz": [70.0, 200.0],
        "hp_order": [2, 4],
        "resonance_db": [2.0, 6.0],
        "lp_hz": [10000.0, 17000.0],
        "lp_order": [2, 4],
        "ripple_db": 4.0,
        "ripples": [2, 4],
        "cavity_prob": 0.3,
        "cavity_hz": [2000.0, 5000.0],
        "cavity_db": [2.0, 5.0],
        "max_spl_db": [84.0, 96.0],
        "level_dba": [56.0, 74.0],
        "thd_full_scale": [0.01, 0.05],
        "virtual_bass_prob": 0.4,
        "dialog_eq_prob": 0.1,
        "drc_prob": 0.6,
        "limiter_prob": 1.0,
        "directivity": _CABINET,
        "content": {"streaming": 0.9, "phone_call": 0.1},
    },
    "radio": {
        "hp_hz": [120.0, 280.0],
        "hp_order": [2, 4],
        "resonance_db": [2.0, 6.0],
        "lp_hz": [5000.0, 11000.0],
        "lp_order": [2, 4],
        "ripple_db": 5.0,
        "ripples": [3, 5],
        "cavity_prob": 0.3,
        "cavity_hz": [1500.0, 4000.0],
        "cavity_db": [2.0, 6.0],
        "max_spl_db": [78.0, 88.0],
        "level_dba": [54.0, 70.0],
        "thd_full_scale": [0.02, 0.06],
        "virtual_bass_prob": 0.1,
        "dialog_eq_prob": 0.0,
        "drc_prob": 0.2,
        "limiter_prob": 0.6,
        "directivity": _CABINET,
        "content": {"radio": 1.0},
    },
    # Door woofers, dash tweeters, coaxial rear speakers: a two-way system whose bands
    # leave from different places in the cabin.
    "car_audio": {
        "hp_hz": [40.0, 90.0],
        "hp_order": [2, 4],
        "resonance_db": [0.5, 4.0],
        "lp_hz": [12000.0, 18000.0],
        "lp_order": [2, 2],
        "ripple_db": 4.0,
        "ripples": [3, 5],
        "cavity_prob": 0.0,
        "cavity_hz": [2000.0, 5000.0],
        "cavity_db": [0.0, 0.0],
        # Set at 1 m: the small cabin adds ~7 dB of reverberant gain on top, and the
        # listener is closer than 1 m to most drivers.
        "max_spl_db": [96.0, 106.0],
        "level_dba": [52.0, 70.0],
        "thd_full_scale": [0.005, 0.02],
        "virtual_bass_prob": 0.0,
        "dialog_eq_prob": 0.2,
        "drc_prob": 0.5,
        "limiter_prob": 0.7,
        "crossover_hz": [2500.0, 4500.0],
        "directivity": [1.0, 1.0, 0.95, 0.85, 0.72, 0.62, 0.56],
        "tweeter_directivity": [1.0, 1.0, 1.0, 0.95, 0.85, 0.72, 0.62],
        "content": {"radio": 0.3, "streaming": 0.35, "phone_call": 0.35},
    },
}

# How much a listener turns the volume up in noise, dB per dB(A) above 45 dB(A).
VOLUME_ONSET_DBA = 45.0


@dataclass
class Device:
    kind: str
    placement: Placement
    driver_filters: list[np.ndarray]
    driver_firs: list[np.ndarray | None]
    directivities: list[Directivity]
    level_dba: float
    max_spl_db: float
    dsp: dict
    content: str
    meta: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "device": self.kind,
            "device_level_dba": round(self.level_dba, 2),
            "device_max_spl_db": round(self.max_spl_db, 2),
            "device_headroom_db": round(self.level_dba - self.max_spl_db, 2),
            "device_dsp": self.dsp,
            "drivers": [p.describe() for p in self.placement.poses],
            **self.meta,
        }


def driver_response_sos(
    sr: int,
    preset: dict,
    rng: np.random.Generator,
    role: str = "full",
    crossover_hz: float | None = None,
) -> tuple[np.ndarray, dict]:
    """IIR model of one driver in its enclosure, as one SOS cascade."""
    if role == "tweeter":
        assert crossover_hz is not None
        lp = min(sample_range(rng, (16000.0, 20000.0)), 0.45 * sr)
        parts = [linkwitz_riley_sos(sr, crossover_hz, "high")]
        for _ in range(2):
            f0 = float(
                np.exp(
                    rng.uniform(
                        np.log(crossover_hz * 1.2), np.log(max(lp * 0.8, crossover_hz * 1.5))
                    )
                )
            )
            parts.append(peaking_sos(sr, f0, rng.uniform(-2.5, 2.5), rng.uniform(1.0, 4.0)))
        if lp < 0.45 * sr:
            parts.append(butter_sos(sr, lp, "low", 2))
        return cascade(*parts), {"role": role, "crossover_hz": round(crossover_hz, 1)}

    hp = sample_range(rng, preset["hp_hz"], log=True)
    lp = sample_range(rng, preset["lp_hz"], log=True)
    hp_order = sample_int(rng, preset["hp_order"])
    lp_order = sample_int(rng, preset["lp_order"])
    parts = [butter_sos(sr, hp, "high", hp_order)]
    res_f = hp * rng.uniform(1.0, 1.4)
    parts.append(
        peaking_sos(sr, res_f, sample_range(rng, preset["resonance_db"]), rng.uniform(1.0, 2.5))
    )
    meta = {"role": role, "hp_hz": round(hp, 1), "hp_order": hp_order, "lp_hz": round(lp, 1)}
    if rng.random() < preset["cavity_prob"]:
        f_cav = sample_range(rng, preset["cavity_hz"], log=True)
        parts.append(
            peaking_sos(sr, f_cav, sample_range(rng, preset["cavity_db"]), rng.uniform(1.5, 4.0))
        )
        meta["cavity_hz"] = round(f_cav, 1)
    for _ in range(sample_int(rng, preset["ripples"])):
        f0 = float(np.exp(rng.uniform(np.log(2.0 * hp), np.log(max(0.7 * lp, 2.5 * hp)))))
        parts.append(
            peaking_sos(
                sr,
                f0,
                rng.uniform(-preset["ripple_db"], preset["ripple_db"]),
                rng.uniform(1.0, 5.0),
            )
        )
    if lp < 0.45 * sr:
        parts.append(butter_sos(sr, lp, "low", lp_order))
    if role == "woofer" and crossover_hz is not None:
        parts.append(linkwitz_riley_sos(sr, crossover_hz, "low"))
        meta["crossover_hz"] = round(crossover_hz, 1)
    return cascade(*parts), meta


def load_measured_response(path: str | Path, sr: int, gate_ms: float = 12.0) -> np.ndarray:
    """A near-field device measurement, gated to its anechoic part and unit-energy.

    Accepts an impulse response ``.wav`` or a model ``.npz`` written by ``extract``.
    Gating after a few milliseconds keeps the driver and cabinet and drops the room
    the measurement was taken in, which the scene supplies itself.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(str(path))
        h = np.asarray(data["g1"], dtype=np.float64)
        h_sr = int(data["sr"]) if "sr" in data else sr
    else:
        h, h_sr = sf.read(str(path), always_2d=False)
        h = to_mono(h)
    h = resample_to(np.asarray(h, dtype=np.float64), h_sr, sr)
    peak = int(np.argmax(np.abs(h)))
    start = max(0, peak - int(0.001 * sr))
    n = int(gate_ms / 1000.0 * sr)
    h = h[start : start + n].copy()
    tail = max(1, len(h) // 4)
    h[-tail:] *= np.hanning(2 * tail)[tail:]
    return h / (np.sqrt(np.sum(h**2)) + EPS)


def sample_device(
    kind: str,
    preset: dict,
    placement: Placement,
    sr: int,
    rng: np.random.Generator,
    noise_dba: float,
    volume_slope,
    content: str,
    measured: list[np.ndarray] | None = None,
) -> Device:
    """Draw one concrete device of type ``kind`` for the given placement."""
    crossover = sample_range(rng, preset["crossover_hz"]) if "crossover_hz" in preset else None
    base_alpha = np.asarray(preset["directivity"], dtype=np.float64).reshape(N_BANDS)
    tweeter_alpha = np.asarray(preset.get("tweeter_directivity", base_alpha), dtype=np.float64)

    # Drivers of the same role are the same part: sample one response per role. A
    # measured near-field response replaces the synthetic model of full-range drivers.
    responses: dict[str, tuple[np.ndarray, dict]] = {}
    fir = measured[int(rng.integers(len(measured)))] if measured else None
    filters, firs, directivities, driver_meta = [], [], [], []
    for pose in placement.poses:
        measured_driver = fir is not None and pose.role == "full"
        if pose.role not in responses:
            if measured_driver:
                responses[pose.role] = (np.zeros((0, 6)), {"role": "full", "measured": True})
            else:
                responses[pose.role] = driver_response_sos(sr, preset, rng, pose.role, crossover)
        sos, meta = responses[pose.role]
        filters.append(sos)
        firs.append(fir if measured_driver else None)
        alpha = tweeter_alpha if pose.role == "tweeter" else base_alpha
        directivities.append(Directivity(pose.axis, alpha))
        driver_meta.append(meta)

    max_spl = sample_range(rng, preset["max_spl_db"])
    slope = sample_range(rng, volume_slope)
    level = sample_range(rng, preset["level_dba"]) + slope * max(0.0, noise_dba - VOLUME_ONSET_DBA)
    level = min(level, max_spl - 1.0)

    dsp: dict = {
        "thd_full_scale": round(sample_range(rng, preset["thd_full_scale"], log=True), 4),
        "h2_share": round(float(rng.uniform(0.2, 0.8)), 3),
        "h2_sign": float(rng.choice([-1.0, 1.0])),
        "excursion_hz": round(
            float(np.median([m.get("hp_hz", 200.0) for m in driver_meta])) * rng.uniform(1.5, 3.0),
            1,
        ),
    }
    if rng.random() < preset["dialog_eq_prob"]:
        dsp["dialog_eq"] = {
            "f0": round(rng.uniform(1500, 3500), 1),
            "gain_db": round(rng.uniform(2, 6), 2),
            "q": round(rng.uniform(0.6, 1.4), 2),
        }
    if rng.random() < preset["virtual_bass_prob"]:
        dsp["virtual_bass"] = {
            "cutoff_hz": round(dsp["excursion_hz"] / 2.0 * rng.uniform(0.9, 1.3), 1),
            "amount_db": round(rng.uniform(-8, -1), 2),
        }
    if rng.random() < preset["drc_prob"]:
        dsp["drc"] = {
            "threshold_rel_db": round(rng.uniform(-4, 6), 2),
            "ratio": round(rng.uniform(1.5, 4), 2),
            "attack_ms": round(rng.uniform(3, 20), 1),
            "release_ms": round(rng.uniform(80, 400), 1),
        }
    if rng.random() < preset["limiter_prob"]:
        dsp["limiter"] = {
            "ceiling_db": round(rng.uniform(-3.0, -0.3), 2),
            "lookahead_ms": round(rng.uniform(1.0, 5.0), 2),
            "release_ms": round(rng.uniform(30, 150), 1),
        }

    return Device(
        kind=kind,
        placement=placement,
        driver_filters=filters,
        driver_firs=firs,
        directivities=directivities,
        level_dba=float(level),
        max_spl_db=float(max_spl),
        dsp=dsp,
        content=content,
        meta={"driver_models": driver_meta, "measured_response": fir is not None, **placement.meta},
    )


def virtual_bass(x: np.ndarray, sr: int, cutoff_hz: float, amount_db: float) -> np.ndarray:
    """Psychoacoustic bass: harmonics of the band a small driver cannot reproduce.

    TVs, phones and laptops generate them so the missing fundamental is still heard
    (Larsen & Aarts, JAES 2002); they survive the driver's high-pass and change the
    harmonic structure of voiced speech in a way no live talker produces.
    """
    low = apply_sos(x, linkwitz_riley_sos(sr, cutoff_hz, "low"))
    peak = float(np.max(np.abs(low))) + EPS
    harm = np.abs(low) + 0.3 * low * np.abs(low) / peak
    harm = apply_sos(harm, butter_sos(sr, (cutoff_hz, min(4.0 * cutoff_hz, 0.45 * sr)), "band", 2))
    gain = rms(low) / (rms(harm) + EPS) * 10.0 ** (amount_db / 20.0)
    return x + gain * harm


def driver_nonlinearity(x: np.ndarray, sr: int, dsp: dict) -> np.ndarray:
    """Excursion-driven polynomial distortion plus saturation at full scale.

    Cone excursion is largest at low frequencies, so the polynomial acts on a
    low-passed copy; ``thd_full_scale`` is the harmonic distortion of a full-scale
    tone in that band, split between second and third harmonic.
    """
    thd = dsp["thd_full_scale"]
    share = dsp["h2_share"]
    exc = apply_sos(x, butter_sos(sr, min(dsp["excursion_hz"], 0.45 * sr), "low", 2))
    a2 = dsp["h2_sign"] * 2.0 * thd * np.sqrt(share)
    a3 = -4.0 * thd * np.sqrt(1.0 - share)
    y = x + a2 * exc**2 + a3 * exc**3
    y = soft_clip(y, knee=0.85)
    return apply_sos(y, butter_sos(sr, 20.0, "high", 2))


def device_emission(program: np.ndarray, sr: int, device: Device) -> tuple[list[np.ndarray], dict]:
    """Per-driver pressure at 1 m on-axis, in pascals."""
    x = np.asarray(program, dtype=np.float64)
    target = device.level_dba - device.max_spl_db + FULL_SCALE_SINE_DB
    x = x * 10.0 ** ((target - speech_level_db(x, sr)) / 20.0)
    dsp = device.dsp
    meta: dict = {}

    if "dialog_eq" in dsp:
        eq = dsp["dialog_eq"]
        x = apply_sos(x, peaking_sos(sr, eq["f0"], eq["gain_db"], eq["q"]))
    if "virtual_bass" in dsp:
        vb = dsp["virtual_bass"]
        x = virtual_bass(x, sr, vb["cutoff_hz"], vb["amount_db"])
    if "drc" in dsp:
        drc = dsp["drc"]
        threshold = active_level_db(x, sr, weighted=False) + drc["threshold_rel_db"]
        x, m = compressor(x, sr, threshold, drc["ratio"], drc["attack_ms"], drc["release_ms"])
        meta["drc_max_reduction_db"] = m["max_reduction_db"]
    if "limiter" in dsp:
        lim = dsp["limiter"]
        x, m = limiter(x, sr, lim["ceiling_db"], lim["lookahead_ms"], lim["release_ms"])
        meta["limiter_max_reduction_db"] = m["max_reduction_db"]
    x = driver_nonlinearity(x, sr, dsp)

    outputs = []
    for sos, fir in zip(device.driver_filters, device.driver_firs, strict=True):
        y = signal.fftconvolve(x, fir)[: len(x)] if fir is not None else x
        outputs.append(apply_sos(y, sos))

    # Calibrate the power sum of all drivers to the target speech level at 1 m.
    hop = max(1, int(0.02 * sr))
    energy = sum(frame_energy(a_weight(y, sr), hop) for y in outputs)
    level = speech_level_from_energy(energy)
    want = 20.0 * np.log10(P_REF) + device.level_dba
    gain = 10.0 ** ((want - level) / 20.0)
    return [y * gain for y in outputs], meta


__all__ = [
    "DEVICE_PRESETS",
    "Device",
    "device_emission",
    "driver_nonlinearity",
    "driver_response_sos",
    "load_measured_response",
    "sample_device",
    "virtual_bass",
]
