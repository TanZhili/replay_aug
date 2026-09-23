"""Source models: live talkers, playback devices and programme chains."""

from __future__ import annotations

import numpy as np
import pytest

from replay_opus.config import Config, TalkerConfig
from replay_opus.content import CONTENT_PROFILES, ContentChain, apply_content, sample_content
from replay_opus.dsp import (
    P_REF,
    a_weight,
    frame_energy,
    sos_response_db,
    speech_level_db,
    speech_level_from_energy,
)
from replay_opus.environment import (
    CAR_DEVICE_LAYOUTS,
    ENVIRONMENTS,
    ROOM_DEVICE_LAYOUTS,
    build_room,
    place_device,
    place_mic,
    place_talker,
)
from replay_opus.loudness import integrated_loudness
from replay_opus.speaker import (
    DEVICE_PRESETS,
    device_emission,
    driver_response_sos,
    sample_device,
    virtual_bass,
)
from replay_opus.talker import sample_talker, talker_emission

SR = 32_000


def _spl(x: np.ndarray) -> float:
    return speech_level_db(x, SR) - 20 * np.log10(P_REF)


def _device(kind: str, env: str, rng, noise_dba: float = 40.0, content: str | None = None):
    preset = ENVIRONMENTS[env]
    room = build_room(preset, rng)
    mic, _ = place_mic(preset, room.dims, rng)
    placement = place_device(kind, preset, room.dims, mic, rng)
    content = content or next(iter(DEVICE_PRESETS[kind]["content"]))
    return sample_device(
        kind, DEVICE_PRESETS[kind], placement, SR, rng, noise_dba, [0.0, 0.0], content
    )


def test_talker_level_is_exact_and_rises_in_noise(rng, speech):
    preset = ENVIRONMENTS["living_room"]
    room = build_room(preset, rng)
    mic, _ = place_mic(preset, room.dims, rng)
    cfg = TalkerConfig(level_dba=60.0, lombard_slope=0.5, lombard_max_db=12.0)
    placement = place_talker(preset, room.dims, mic, rng)
    quiet = sample_talker(placement, cfg, 35.0, rng)
    loud = sample_talker(placement, cfg, 70.0, rng)
    assert quiet.level_dba == pytest.approx(60.0)
    assert loud.lombard_db == pytest.approx(12.0)  # 0.5 * (70 - 45) = 12.5, capped
    assert _spl(talker_emission(speech, SR, loud)) == pytest.approx(72.0, abs=0.05)
    assert (
        loud.tilt_db > 0 > sample_talker(placement, TalkerConfig(level_dba=52.0), 30.0, rng).tilt_db
    )


def test_device_output_is_calibrated_to_its_level(rng, speech):
    device = _device("tv", "living_room", rng)
    drivers, _ = device_emission(speech, SR, device)
    hop = int(0.02 * SR)
    energy = sum(frame_energy(a_weight(d, SR), hop) for d in drivers)
    level = speech_level_from_energy(energy) - 20 * np.log10(P_REF)
    assert level == pytest.approx(device.level_dba, abs=0.05)
    assert len(drivers) == len(device.placement.poses) == 2


def test_small_drivers_lose_the_low_end(rng):
    sos, meta = driver_response_sos(SR, DEVICE_PRESETS["phone"], rng)
    lows, mids = sos_response_db(sos, SR, np.array([100.0, 2000.0]))
    assert mids - lows > 25.0
    assert 400.0 <= meta["hp_hz"] <= 900.0


def test_distortion_grows_with_drive(speech):
    residuals = []
    for headroom in (-40.0, -3.0):
        device = _device("phone", "living_room", np.random.default_rng(3))
        device.max_spl_db = device.level_dba - headroom
        device.dsp.pop("limiter", None)
        device.dsp["thd_full_scale"] = 0.08
        out = device_emission(speech, SR, device)[0][0]
        clean = device_emission(speech * 1e-3, SR, device)[0][0]
        gain = np.dot(out, clean) / np.dot(clean, clean)
        residuals.append(10 * np.log10(np.sum((out - gain * clean) ** 2) / np.sum(out**2)))
    assert residuals[1] > residuals[0] + 10.0


def test_virtual_bass_adds_harmonics_above_its_cutoff():
    t = np.arange(SR) / SR
    tone = 0.3 * np.sin(2 * np.pi * 100 * t)
    out = virtual_bass(tone, SR, cutoff_hz=150.0, amount_db=-3.0)
    spectrum = np.abs(np.fft.rfft(out - tone))
    freqs = np.fft.rfftfreq(len(t), 1 / SR)
    assert spectrum[np.argmin(np.abs(freqs - 200))] > 10 * spectrum[np.argmin(np.abs(freqs - 1000))]


@pytest.mark.parametrize("profile", list(CONTENT_PROFILES))
def test_content_chains_run_and_land_near_their_loudness(profile, speech):
    rng = np.random.default_rng(1)
    chain = sample_content(profile, CONTENT_PROFILES[profile], rng, set(), has_beds=False)
    y, meta = apply_content(speech, SR, chain, rng)
    assert np.all(np.isfinite(y))
    assert meta["content"] == profile
    lo, hi = CONTENT_PROFILES[profile]["loudness_lufs"]
    assert lo - 3.0 <= integrated_loudness(y, SR) <= hi + 1.0


def test_narrowband_call_has_no_high_band(speech):
    chain = ContentChain(
        "phone_call",
        {
            "loudness_lufs": -20.0,
            "band": "narrow",
            "limiter_ceiling_db": -1.0,
            "codec": "pcm_alaw",
            "bitrate_kbps": 64,
            "codec_rate": 8000,
        },
    )
    y, _ = apply_content(speech, SR, chain, np.random.default_rng(0))
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    assert 10 * np.log10(spectrum[freqs > 4500].sum() / spectrum.sum()) < -35.0


@pytest.mark.parametrize(
    ("kind", "env"),
    [(k, "living_room") for k in ROOM_DEVICE_LAYOUTS]
    + [(k, "car_cabin") for k in CAR_DEVICE_LAYOUTS],
)
def test_every_device_type_renders(kind, env, speech):
    rng = np.random.default_rng(5)
    device = _device(kind, env, rng)
    drivers, _ = device_emission(speech, SR, device)
    assert all(np.all(np.isfinite(d)) for d in drivers)
    for pose in device.placement.poses:
        assert np.all(pose.position > 0)
    if kind == "car_audio":
        assert {"woofer", "tweeter"} <= {p.role for p in device.placement.poses}


def test_default_config_is_consistent():
    Config().validate()
