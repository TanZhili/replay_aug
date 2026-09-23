"""Property tests for the replay chain.

These assert behaviour that must hold regardless of implementation, not snapshots.
Run with ``python -m pytest tests -q`` (or ``python -m replay_opus selftest``).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from replay_opus.capture import adc_clip, simple_agc
from replay_opus.chain import ReplayChain
from replay_opus.config import Config
from replay_opus.dsp import align_to, apply_clock_drift, set_peak_db
from replay_opus.irbank import reshape_ir, synth_ir
from replay_opus.measure import (
    SweepSpec,
    extract_linear_ir,
    generate_sweep,
    harmonic_profile,
)
from replay_opus.metrics import describe_ir, estimate_drr, estimate_t60
from replay_opus.noise import mix_at_snr, synth_noise
from replay_opus.transport import CodecSpec, available_codecs, codec_roundtrip

SR = 48_000


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.mark.parametrize("t60,drr", [(0.12, 18.0), (0.4, 8.0), (1.0, 0.0)])
def test_synth_ir_hits_targets(rng, t60, drr):
    got = describe_ir(synth_ir(SR, t60, drr, rng), SR)
    assert got["t60_s"] == pytest.approx(t60, rel=0.1)
    assert got["drr_db"] == pytest.approx(drr, abs=0.5)


def test_reshape_shortens_t60_and_sets_drr(rng):
    base = synth_ir(SR, 0.9, 4.0, rng)
    out = reshape_ir(base, SR, 0.25, 16.0)
    assert estimate_t60(out, SR) == pytest.approx(0.25, rel=0.12)
    assert estimate_drr(out, SR) == pytest.approx(16.0, abs=0.5)


def test_reshape_cannot_invent_a_longer_tail(rng):
    base = synth_ir(SR, 0.3, 6.0, rng)
    out = reshape_ir(base, SR, 2.0, 6.0)
    assert estimate_t60(out, SR) <= estimate_t60(base, SR) * 1.1


def test_measured_ir_reproduces_the_system(rng):
    spec = SweepSpec(sr=SR)
    sweep, _, _ = generate_sweep(spec)
    room = synth_ir(SR, 0.4, 9.0, rng)
    measured, _ = extract_linear_ir(signal.fftconvolve(sweep, room), spec, ir_ms=1500)

    probe = signal.sosfilt(
        signal.butter(4, (200 / (SR / 2), 6000 / (SR / 2)), btype="band", output="sos"),
        rng.standard_normal(SR),
    )
    truth = signal.fftconvolve(probe, room)
    pred, _ = align_to(signal.fftconvolve(probe, measured), truth)
    n = min(len(pred), len(truth))
    gain = np.dot(truth[:n], pred[:n]) / np.dot(pred[:n], pred[:n])
    residual = gain * pred[:n] - truth[:n]
    rel_db = 20 * np.log10(np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(truth[:n] ** 2)))
    assert rel_db < -30.0


@pytest.mark.parametrize("c2,c3", [(0.3, 0.1), (0.05, 0.02)])
def test_harmonic_profile_matches_theory(c2, c3):
    spec = SweepSpec(sr=SR)
    sweep, _, _ = generate_sweep(spec)
    profile = harmonic_profile(sweep + c2 * sweep**2 + c3 * sweep**3, spec, [1, 2, 3, 4])
    fundamental = 1.0 + 0.75 * c3
    assert profile["level_db"][1] == pytest.approx(20 * np.log10(0.5 * c2 / fundamental), abs=0.2)
    assert profile["level_db"][2] == pytest.approx(20 * np.log10(0.25 * c3 / fundamental), abs=0.2)


@pytest.mark.parametrize("codec", available_codecs())
def test_codec_roundtrip_preserves_length_and_alignment(rng, codec):
    x = 0.4 * signal.sosfilt(
        signal.butter(4, (200 / (SR / 2), 6000 / (SR / 2)), btype="band", output="sos"),
        rng.standard_normal(SR),
    )
    y, _ = codec_roundtrip(x, SR, CodecSpec(codec))
    assert len(y) == len(x)
    # The decoder's own priming delay must be removed, or every file would carry a
    # codec-dependent latency.
    assert abs(align_to(y, x)[1]) <= 2


def test_mix_at_snr_produces_the_requested_ratio(rng):
    speech = rng.standard_normal(SR)
    noise = synth_noise("pink", SR, SR, rng)
    for target in (5.0, 20.0, 35.0):
        mixed = mix_at_snr(speech, noise, target)
        residual = mixed - speech
        got = 20 * np.log10(np.sqrt(np.mean(speech**2)) / np.sqrt(np.mean(residual**2)))
        assert got == pytest.approx(target, abs=0.5)


def test_agc_settles_a_level_step(rng):
    """A sustained step should be compressed once the tracker has had time to follow.

    The tracker recovers at ``level_decay_db_s`` (6 dB/s by default), so the settled
    level is measured several seconds in rather than immediately after the step.
    """
    loud = 0.5 * rng.standard_normal(2 * SR)
    quiet = 0.05 * rng.standard_normal(4 * SR)
    gained, _ = simple_agc(np.concatenate([loud, quiet]), SR, target_dbfs=-20.0)

    def level_db(segment: np.ndarray) -> float:
        return 20 * np.log10(np.std(segment))

    step_in = level_db(loud) - level_db(quiet)
    step_out = level_db(gained[SR // 2 : 2 * SR]) - level_db(gained[5 * SR : 6 * SR])
    assert step_in > 15.0
    assert step_out < 5.0


def test_agc_does_not_lift_a_pause_above_the_speech(rng):
    """The failure this guards against: gain chasing frame level, not speech level."""
    speech = 0.3 * rng.standard_normal(2 * SR)
    pause = 0.0005 * rng.standard_normal(2 * SR)
    gained, meta = simple_agc(np.concatenate([speech, pause]), SR, target_dbfs=-20.0)
    speech_out = 20 * np.log10(np.std(gained[: 2 * SR]))
    pause_out = 20 * np.log10(np.std(gained[2 * SR :]))
    assert pause_out < speech_out - 20.0
    assert meta["agc_gain_db_range"][1] - meta["agc_gain_db_range"][0] < 25.0


def test_adc_clip_reports_clipping_only_when_driven_over(rng):
    x = rng.standard_normal(SR)
    _, quiet = adc_clip(x, -6.0)
    _, hot = adc_clip(x, 3.0)
    assert quiet["clipped_samples"] == 0
    assert hot["clipped_samples"] > 0


def test_clock_drift_changes_duration_by_the_requested_amount():
    x = np.zeros(SR)
    x[::100] = 1.0
    assert len(apply_clock_drift(x, 100.0)) == pytest.approx(SR * 1.0001, abs=1)
    assert len(apply_clock_drift(x, -100.0)) == pytest.approx(SR * 0.9999, abs=1)


def test_set_peak_db_is_exact(rng):
    x = rng.standard_normal(1000)
    for target in (-20.0, -3.0, 0.0):
        assert 20 * np.log10(np.max(np.abs(set_peak_db(x, target)))) == pytest.approx(
            target, abs=1e-6
        )


def test_chain_is_deterministic_and_variant_dependent(rng):
    chain = ReplayChain(Config())
    source = 0.4 * signal.sosfilt(
        signal.butter(4, (200 / (SR / 2), 4000 / (SR / 2)), btype="band", output="sos"),
        rng.standard_normal(2 * SR),
    )
    first = chain.process(source, SR, key="a", variant=0)
    assert np.array_equal(first.audio, chain.process(source, SR, key="a", variant=0).audio)

    other = chain.process(source, SR, key="a", variant=1)
    n = min(len(first.audio), len(other.audio))
    assert not np.array_equal(first.audio[:n], other.audio[:n])

    assert first.sr == chain.config.output.sr
    assert np.max(np.abs(first.audio)) <= 1.0


@pytest.mark.parametrize("tail_ms", [0.0, 200.0, 600.0])
def test_chain_bounds_the_reverberant_tail(rng, tail_ms):
    config = Config()
    config.output.tail_ms = tail_ms
    config.room.t60_s = 0.9  # a long IR, so the tail cap is what limits the length
    chain = ReplayChain(config)
    source = 0.3 * rng.standard_normal(2 * SR)

    result = chain.process(source, SR, key="a", variant=0)
    expected = 2.0 + tail_ms / 1000.0
    assert result.meta["duration_s"] == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("mode", ["synthetic", "simulated"])
def test_room_modes_produce_usable_irs(rng, mode):
    config = Config()
    config.room.mode = mode
    chain = ReplayChain(config)
    meta = chain.process(0.3 * rng.standard_normal(SR), SR, key="a", variant=0).meta
    measured = meta["room_ir_measured"]
    assert 0.02 < measured["t60_s"] < 3.0
    assert -20.0 < measured["drr_db"] < 40.0


def test_simulated_rooms_separate_source_and_mic(rng):
    """Independent placement used to allow a zero-distance, degenerate geometry."""
    config = Config()
    config.room.mode = "simulated"
    chain = ReplayChain(config)
    for variant in range(8):
        meta = chain.process(0.3 * rng.standard_normal(SR // 2), SR, key="a", variant=variant).meta
        assert meta["room_distance_m"] >= 0.2


def test_chain_metadata_records_every_sampled_condition(rng):
    chain = ReplayChain(Config())
    source = 0.3 * rng.standard_normal(SR)
    meta = chain.process(source, SR, key="a", variant=0).meta
    for field in (
        "speaker",
        "drive_db",
        "room_mode",
        "room_ir_measured",
        "snr_db",
        "mic",
        "adc_headroom_db",
        "clock_drift_ppm",
        "codec",
        "out_sr",
    ):
        assert field in meta, f"{field} missing from manifest metadata"


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown config key"):
        Config.from_dict({"room": {"nope": 1}})


def test_config_requires_ir_list_for_measured_mode():
    config = Config.from_dict({"room": {"mode": "measured"}})
    with pytest.raises(ValueError, match=r"requires room\.ir_list"):
        config.validate()
