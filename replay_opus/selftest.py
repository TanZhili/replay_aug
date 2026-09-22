"""End-to-end checks against systems whose answers are known analytically.

Run with ``python -m replay_opus selftest``. Every assertion here is a property the
pipeline must hold, not a regression snapshot, so the thresholds stay meaningful if
the implementation changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .chain import ReplayChain
from .config import Config
from .dsp import align_to
from .irbank import reshape_ir, synth_ir
from .measure import SweepSpec, extract_linear_ir, generate_sweep, harmonic_profile
from .metrics import describe_ir
from .transport import CodecSpec, available_codecs, codec_roundtrip

SR = 48_000


def _residual_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    n = min(len(estimate), len(reference))
    a, b = estimate[:n], reference[:n]
    gain = float(np.dot(b, a) / (np.dot(a, a) + 1e-30))
    residual = gain * a - b
    return float(
        20.0 * np.log10(np.sqrt(np.mean(residual**2)) / (np.sqrt(np.mean(b**2)) + 1e-30) + 1e-15)
    )


def _speech_like(rng: np.random.Generator, seconds: float = 3.0) -> np.ndarray:
    """A harmonic-plus-noise signal with a speech-shaped spectrum and syllabic envelope."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    f0 = 120.0 + 20.0 * np.sin(2.0 * np.pi * 1.7 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / SR
    excitation = sum(np.sin(k * phase) / k for k in range(1, 25))
    excitation += 0.15 * rng.standard_normal(n)
    sos = signal.butter(2, (200.0 / (SR / 2), 3500.0 / (SR / 2)), btype="band", output="sos")
    shaped = signal.sosfilt(sos, excitation)
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t) ** 2
    out = shaped * envelope
    return 0.5 * out / (np.max(np.abs(out)) + 1e-12)


def run_selftest(output_dir: Path | None = None) -> int:
    rng = np.random.default_rng(0)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print("IR synthesis")
    for t60, drr in ((0.15, 15.0), (0.5, 5.0), (1.2, 0.0)):
        got = describe_ir(synth_ir(SR, t60, drr, rng), SR)
        ok = abs(got["t60_s"] - t60) / t60 < 0.1 and abs(got["drr_db"] - drr) < 0.5
        check(
            f"synth_ir T60={t60} DRR={drr}",
            ok,
            f"got T60={got['t60_s']:.3f} DRR={got['drr_db']:.2f}",
        )

    print("IR retargeting")
    base = synth_ir(SR, 0.9, 5.0, rng)
    got = describe_ir(reshape_ir(base, SR, 0.3, 18.0), SR)
    check(
        "reshape_ir to T60=0.30 DRR=18",
        abs(got["t60_s"] - 0.3) / 0.3 < 0.1 and abs(got["drr_db"] - 18.0) < 0.5,
        f"got T60={got['t60_s']:.3f} DRR={got['drr_db']:.2f}",
    )

    print("Sweep measurement")
    spec = SweepSpec(sr=SR)
    sweep, _, _ = generate_sweep(spec)
    room = synth_ir(SR, 0.45, 8.0, rng)
    measured, _ = extract_linear_ir(signal.fftconvolve(sweep, room), spec, ir_ms=1500)
    got, truth = describe_ir(measured, SR), describe_ir(room, SR)
    check(
        "room IR recovered from a sweep",
        abs(got["t60_s"] - truth["t60_s"]) < 0.02 and abs(got["drr_db"] - truth["drr_db"]) < 0.5,
        f"T60 {truth['t60_s']:.3f}->{got['t60_s']:.3f}, "
        f"DRR {truth['drr_db']:.2f}->{got['drr_db']:.2f}",
    )

    probe = _speech_like(rng)
    predicted, _ = align_to(signal.fftconvolve(probe, measured), signal.fftconvolve(probe, room))
    residual = _residual_db(predicted, signal.fftconvolve(probe, room))
    check("measured IR reproduces the system", residual < -30.0, f"{residual:.1f} dB residual")

    c2, c3 = 0.30, 0.10
    profile = harmonic_profile(sweep + c2 * sweep**2 + c3 * sweep**3, spec, [1, 2, 3, 4])
    want2 = 20.0 * np.log10(0.5 * c2 / (1 + 0.75 * c3))
    want3 = 20.0 * np.log10(0.25 * c3 / (1 + 0.75 * c3))
    check(
        "harmonic profile of a known polynomial",
        abs(profile["level_db"][1] - want2) < 0.2 and abs(profile["level_db"][2] - want3) < 0.2,
        f"H2 {profile['level_db'][1]:.2f} (want {want2:.2f}), "
        f"H3 {profile['level_db'][2]:.2f} (want {want3:.2f})",
    )

    print("Codecs")
    tone = _speech_like(rng, 2.0)
    # AMR-NB is a parametric narrowband vocoder, not a waveform coder, so it is held
    # to a far looser level tolerance than the transform codecs.
    tolerances = {"amr_nb": (0.3, 1.6)}
    for name in available_codecs():
        decoded, meta = codec_roundtrip(tone, SR, CodecSpec(name))
        gain = float(np.dot(tone, decoded[: len(tone)]) / np.dot(tone, tone))
        low, high = tolerances.get(name, (0.9, 1.1))
        check(
            f"{name} round trip",
            len(decoded) == len(tone) and low < gain < high,
            f"gain={gain:.3f}, delay={meta['codec_delay_samples']} samples compensated",
        )

    print("Full chain")
    config = Config()
    config.room.mode = "synthetic"
    chain = ReplayChain(config)
    source = _speech_like(rng, 3.0)
    result = chain.process(source, SR, key="selftest", variant=0)
    check(
        "chain produces audio at the configured rate",
        result.sr == config.output.sr and len(result.audio) > SR,
        f"{len(result.audio)} samples @ {result.sr} Hz",
    )
    check(
        "chain output is not clipped",
        float(np.max(np.abs(result.audio))) <= 1.0,
        f"peak {20 * np.log10(np.max(np.abs(result.audio)) + 1e-12):.2f} dBFS",
    )
    repeat = chain.process(source, SR, key="selftest", variant=0)
    check(
        "same seed reproduces the same output",
        bool(np.array_equal(result.audio, repeat.audio)),
        "bit-identical on re-run",
    )
    varied = chain.process(source, SR, key="selftest", variant=1)
    check(
        "different variant gives different conditions",
        not np.array_equal(result.audio[: len(varied.audio)], varied.audio[: len(result.audio)]),
        "variant 1 differs from variant 0",
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(output_dir / "source.wav", source, SR, subtype="PCM_16")
        for variant in range(4):
            out = chain.process(source, SR, key="selftest", variant=variant)
            sf.write(output_dir / f"replayed_v{variant}.wav", out.audio, out.sr, subtype="PCM_16")
        print(f"\ndemo audio written to {output_dir}")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0
