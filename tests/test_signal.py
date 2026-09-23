"""Signal basics: weighting, levels, loudness, dynamics and codecs."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from replay_opus.codecs import CodecSpec, available_codecs, codec_roundtrip, encoder_rate
from replay_opus.dsp import (
    a_weighting_sos,
    active_level_db,
    align_to,
    level_dba,
    sos_response_db,
    speech_segments,
)
from replay_opus.dynamics import agc, compressor, limiter, multiband_compressor
from replay_opus.loudness import integrated_loudness, normalize_loudness

SR = 32_000


@pytest.mark.parametrize("sr", [48_000, 32_000])
def test_a_weighting_matches_the_standard(sr):
    got = sos_response_db(a_weighting_sos(sr), sr, np.array([100.0, 1000.0, 2000.0]))
    assert got == pytest.approx([-19.1, 0.0, 1.2], abs=0.3)


def test_level_of_a_94_db_tone():
    tone = np.sqrt(2.0) * np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)  # 1 Pa rms
    assert level_dba(tone, SR) == pytest.approx(94.0, abs=0.1)


@pytest.mark.parametrize("sr", [48_000, 32_000])
def test_full_scale_sine_reads_minus_3_lufs(sr):
    t = np.arange(3 * sr) / sr
    assert integrated_loudness(np.sin(2 * np.pi * 997 * t), sr) == pytest.approx(-3.01, abs=0.05)


def test_normalize_loudness_hits_target(speech):
    y, _ = normalize_loudness(speech, SR, -23.0)
    assert integrated_loudness(y, SR) == pytest.approx(-23.0, abs=0.05)


def test_active_level_ignores_silence(speech):
    padded = np.concatenate([speech, np.zeros(3 * SR)])
    assert active_level_db(padded, SR) == pytest.approx(active_level_db(speech, SR), abs=0.3)
    rms = lambda x: 20 * np.log10(np.sqrt(np.mean(x**2)))  # noqa: E731
    assert rms(padded) < rms(speech) - 3.0


def test_speech_segments_find_the_active_part(speech):
    segments = speech_segments(speech, SR)
    assert len(segments) == 1
    start, end = segments[0]
    assert start == pytest.approx(0.3, abs=0.05)
    assert end == pytest.approx(1.5, abs=0.08)


def test_compressor_reaches_the_static_curve():
    t = np.arange(4 * SR) / SR
    tone = np.sqrt(2) * 10 ** (-10 / 20) * np.sin(2 * np.pi * 440 * t)
    y, _ = compressor(tone, SR, threshold_db=-20.0, ratio=4.0, knee_db=0.0)
    level = 20 * np.log10(np.sqrt(np.mean(y[SR:] ** 2)))
    assert level == pytest.approx(-17.5, abs=0.6)


def test_limiter_holds_its_ceiling(rng):
    t = np.arange(3 * SR) / SR
    x = rng.standard_normal(len(t)) * (0.2 + 0.8 * (np.sin(2 * np.pi * 2 * t) > 0))
    y, meta = limiter(x, SR, ceiling_db=-6.0)
    assert np.max(np.abs(y)) <= 10 ** (-6 / 20) + 1e-12
    assert meta["max_reduction_db"] > 6.0
    quiet = 0.01 * rng.standard_normal(SR)
    assert np.allclose(limiter(quiet, SR, ceiling_db=-1.0)[0], quiet)


def test_multiband_with_unit_ratio_is_transparent(rng):
    x = rng.standard_normal(SR)
    y, _ = multiband_compressor(x, SR, (250.0, 2500.0), threshold_db=0.0, ratio=1.0)
    assert np.max(np.abs(y - x)) < 1e-9


def test_agc_settles_a_level_step(rng):
    loud = 0.5 * rng.standard_normal(2 * SR)
    quiet = 0.05 * rng.standard_normal(4 * SR)
    gained, _ = agc(np.concatenate([loud, quiet]), SR, target_dbfs=-20.0)
    level = lambda s: 20 * np.log10(np.std(s))  # noqa: E731
    assert level(loud) - level(quiet) > 15.0
    assert level(gained[SR // 2 : 2 * SR]) - level(gained[5 * SR : 6 * SR]) < 5.0


def test_agc_does_not_lift_a_pause_above_the_speech(rng):
    speech = 0.3 * rng.standard_normal(2 * SR)
    pause = 0.0005 * rng.standard_normal(2 * SR)
    gained, _ = agc(np.concatenate([speech, pause]), SR, target_dbfs=-20.0)
    assert 20 * np.log10(np.std(gained[2 * SR :])) < 20 * np.log10(np.std(gained[: 2 * SR])) - 20.0


def test_agc_starts_settled(rng):
    """A warm start: no gain ramp over the first half second."""
    x = 0.01 * rng.standard_normal(3 * SR)
    gained, _ = agc(x, SR, target_dbfs=-20.0)
    first = 20 * np.log10(np.std(gained[: SR // 10]))
    later = 20 * np.log10(np.std(gained[2 * SR :]))
    assert abs(first - later) < 1.5


@pytest.mark.parametrize("codec", available_codecs())
def test_codec_roundtrip_keeps_length_and_alignment(rng, codec):
    sos = signal.butter(4, (200 / (SR / 2), 3400 / (SR / 2)), btype="band", output="sos")
    x = 0.3 * signal.sosfilt(sos, rng.standard_normal(SR))
    y, meta = codec_roundtrip(x, SR, CodecSpec(codec))
    assert len(y) == len(x)
    assert abs(align_to(y, x)[1]) <= 2
    assert meta["codec_sr"] in CodecSpec(codec).info().rates


def test_codec_bitrates_snap_to_valid_modes():
    assert CodecSpec("amr_nb", 9000).resolved_bitrate() == 7950
    assert CodecSpec("ac3", 200_000).resolved_bitrate() == 192_000
    assert CodecSpec("opus", 23_456).resolved_bitrate() == 23_456


def test_encoder_rate_prefers_rates_at_or_above_the_signal():
    assert encoder_rate(32_000, (48_000, 24_000, 16_000)) == 48_000
    assert encoder_rate(32_000, (48_000, 24_000), preferred=24_000) == 24_000
    assert encoder_rate(96_000, (48_000, 44_100)) == 48_000
