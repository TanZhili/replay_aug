"""Swept-sine measurement of real devices, against systems with known answers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from replay_opus.dsp import align_to
from replay_opus.irbank import synth_ir
from replay_opus.measure import (
    SweepSpec,
    extract_linear_ir,
    generate_sweep,
    harmonic_profile,
    measure_chain,
)
from replay_opus.speaker import load_measured_response

SR = 48_000


def test_measured_ir_reproduces_the_system(rng):
    spec = SweepSpec(sr=SR)
    sweep, _, _ = generate_sweep(spec)
    system = synth_ir(SR, 0.4, 9.0, rng)
    measured, _ = extract_linear_ir(signal.fftconvolve(sweep, system), spec, ir_ms=1500)
    probe = signal.sosfilt(
        signal.butter(4, (200 / (SR / 2), 6000 / (SR / 2)), btype="band", output="sos"),
        rng.standard_normal(SR),
    )
    truth = signal.fftconvolve(probe, system)
    pred, _ = align_to(signal.fftconvolve(probe, measured), truth)
    n = min(len(pred), len(truth))
    gain = np.dot(truth[:n], pred[:n]) / np.dot(pred[:n], pred[:n])
    residual = gain * pred[:n] - truth[:n]
    assert 20 * np.log10(np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(truth[:n] ** 2))) < -30.0


@pytest.mark.parametrize(("c2", "c3"), [(0.3, 0.1), (0.05, 0.02)])
def test_harmonic_profile_matches_theory(c2, c3):
    spec = SweepSpec(sr=SR)
    sweep, _, _ = generate_sweep(spec)
    profile = harmonic_profile(sweep + c2 * sweep**2 + c3 * sweep**3, spec, [1, 2, 3, 4])
    fundamental = 1.0 + 0.75 * c3
    assert profile["level_db"][1] == pytest.approx(20 * np.log10(0.5 * c2 / fundamental), abs=0.2)
    assert profile["level_db"][2] == pytest.approx(20 * np.log10(0.25 * c3 / fundamental), abs=0.2)


def test_measure_chain_reports_distortion_and_feeds_the_device_loader(tmp_path):
    spec = SweepSpec(sr=SR)
    sweep, _, _ = generate_sweep(spec)
    ir, info = measure_chain(sweep + 0.1 * sweep**2, spec)
    assert info["thd_at_sweep_level"] == pytest.approx(0.05, rel=0.1)
    path = tmp_path / "device.npz"
    np.savez(path, g1=ir[:4096], sr=SR)
    response = load_measured_response(path, 32_000)
    assert np.sum(response**2) == pytest.approx(1.0)
    assert len(response) == int(0.012 * 32_000)
