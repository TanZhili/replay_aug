"""End-to-end checks against answers known in advance.

Run with ``python -m replay_opus selftest``. Every check is a property the pipeline
must hold - a physical law or an invariant between the classes - not a snapshot, so
the thresholds stay meaningful if the implementation changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .acoustics import SPEED_OF_SOUND, Room, absorption_for_rt60, room_impulse_response
from .codecs import CodecSpec, available_codecs, codec_roundtrip
from .compose import SceneComposer
from .config import Config
from .dsp import align_to, level_dba
from .irbank import synth_ir
from .loudness import integrated_loudness
from .measure import SweepSpec, extract_linear_ir, generate_sweep, harmonic_profile
from .metrics import estimate_drr, estimate_t60
from .pathlist import Utterance
from .render import UtteranceRenderer

SR = 32_000


def _speech_like(rng: np.random.Generator, seconds: float = 1.5, lead: float = 0.3) -> np.ndarray:
    """Harmonic bursts with a speech-shaped spectrum, framed by silence."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    phase = 2 * np.pi * np.cumsum(140.0 * (1 + 0.08 * np.sin(2 * np.pi * 1.3 * t))) / SR
    voiced = sum(np.sin(k * phase) / k for k in range(1, 40)) + 0.1 * rng.standard_normal(n)
    sos = signal.butter(2, (150 / (SR / 2), 5000 / (SR / 2)), btype="band", output="sos")
    burst = signal.sosfilt(sos, voiced) * (0.3 + 0.7 * np.sin(2 * np.pi * 3.5 * t) ** 2)
    burst = 0.5 * burst / np.max(np.abs(burst))
    return np.concatenate([np.zeros(int(SR * lead)), burst, np.zeros(int(SR * 0.2))])


def _band(h: np.ndarray, fc: float) -> np.ndarray:
    sos = signal.butter(3, (fc / 1.41 / (SR / 2), fc * 1.41 / (SR / 2)), btype="band", output="sos")
    return signal.sosfiltfilt(sos, h)


def run_selftest(output_dir: Path | None = None) -> int:
    rng = np.random.default_rng(0)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print("Levels")
    t = np.arange(3 * 48_000) / 48_000
    lufs = integrated_loudness(np.sin(2 * np.pi * 997 * t), 48_000)
    check("BS.1770 full-scale sine", abs(lufs + 3.01) < 0.05, f"{lufs:.2f} LUFS (want -3.01)")
    tone = np.sqrt(2) * np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)
    dba = level_dba(tone, SR)
    check("A-weighted 1 Pa tone", abs(dba - 94.0) < 0.1, f"{dba:.2f} dB(A) (want 94.0)")

    print("Room acoustics")
    dims = np.array([5.5, 4.2, 2.7])
    room = Room(dims, absorption_for_rt60(dims, np.full(7, 0.55), np.array([1, 1, 1, 1, 1.5, 0.8])))
    src = np.array([1.0, 2.0, 1.2])
    far = [
        room_impulse_response(room, src, [4.2, 2.6, 1.0], SR, np.random.default_rng(s))[0]
        for s in range(3)
    ]
    t30 = float(np.median([estimate_t60(_band(h, 1000.0), SR) for h in far]))
    want = float(room.rt60()[3])
    check(
        "RT60 at 1 kHz follows Eyring",
        abs(t30 - want) / want < 0.12,
        f"{t30:.3f} s (want {want:.3f})",
    )
    near = [
        room_impulse_response(room, src, src + np.array([1.0, 0, 0]), SR, np.random.default_rng(s))[
            0
        ]
        for s in range(3)
    ]
    drr = float(np.mean([estimate_drr(h, SR) for h in near]))
    theory = 10 * np.log10(
        1.0 / (312.2 * want / room.volume * np.exp(-13.82 / SPEED_OF_SOUND / want))
    )
    check(
        "DRR at 1 m follows diffuse theory",
        abs(drr - theory) < 2.0,
        f"{drr:.2f} dB (theory {theory:.2f})",
    )

    print("Codecs")
    burst = _speech_like(rng, 1.0, 0.0)
    for name in available_codecs():
        decoded, meta = codec_roundtrip(burst, SR, CodecSpec(name))
        lag = align_to(decoded, burst)[1]
        check(
            f"{name} round trip",
            len(decoded) == len(burst) and abs(lag) <= 2,
            f"{meta['codec_sr']} Hz, residual lag {lag} samples",
        )

    print("Device measurement")
    spec = SweepSpec(sr=48_000)
    sweep, _, _ = generate_sweep(spec)
    system = synth_ir(48_000, 0.3, 8.0, rng)
    measured, _ = extract_linear_ir(signal.fftconvolve(sweep, system), spec, ir_ms=1000)
    # Probe inside the sweep's band: outside it the measurement has nothing to recover.
    band = signal.butter(4, (200 / 24_000, 6000 / 24_000), btype="band", output="sos")
    probe = signal.sosfilt(band, rng.standard_normal(48_000))
    truth = signal.fftconvolve(probe, system)
    pred, _ = align_to(signal.fftconvolve(probe, measured), truth)
    n = min(len(pred), len(truth))
    gain = np.dot(truth[:n], pred[:n]) / np.dot(pred[:n], pred[:n])
    residual = 20 * np.log10(np.std(gain * pred[:n] - truth[:n]) / np.std(truth[:n]))
    check("sweep recovers the system", residual < -30.0, f"{residual:.1f} dB residual")
    profile = harmonic_profile(sweep + 0.3 * sweep**2, spec, [1, 2, 3])
    want_h2 = 20 * np.log10(0.15)
    check(
        "second harmonic of a known polynomial",
        abs(profile["level_db"][1] - want_h2) < 0.2,
        f"{profile['level_db'][1]:.2f} dB (want {want_h2:.2f})",
    )

    print("Paired rendering")
    config = Config()
    config.capture.agc_prob = 0.0
    config.content.weights = {"broadcast": 1.0}
    config.content.profiles = {"broadcast": {"codecs": {"none": 1.0}, "source_room_prob": 0.0}}
    renderer = UtteranceRenderer(config)
    speech = _speech_like(rng)
    live, device = renderer.render(speech, SR, "selftest", 0)
    check(
        "both classes rendered",
        (live.label, device.label) == ("live", "device"),
        f"{live.meta['scene']['environment']}, {device.meta['source']['device']}",
    )
    check(
        "same scene, same length",
        live.meta["scene"] == device.meta["scene"] and len(live.audio) == len(device.audio),
        f"{len(live.audio)} samples @ {live.sr} Hz",
    )
    lead = int(0.2 * live.sr)
    same = float(np.max(np.abs(live.audio[:lead] - device.audio[:lead])))
    check("identical noise before the speech", same < 1e-6, f"max difference {same:.1e}")
    again = renderer.render(speech, SR, "selftest", 0)
    check(
        "same seed, same output",
        all(np.array_equal(a.audio, b.audio) for a, b in zip((live, device), again, strict=True)),
        "bit-identical",
    )

    print("Full-duplex scenes")
    scratch = Path(output_dir) if output_dir else None
    utt_dir = scratch or Path.cwd()
    utt_path = utt_dir / "_selftest_utt.wav"
    sf.write(utt_path, speech, SR)
    try:
        scene_config = Config()
        scene_config.scene.duration_s = [6.0, 6.0]
        scene_config.scene.types = {"both": 1.0}
        scene_config.scene.save_stems = True
        scene = SceneComposer(scene_config, [Utterance(utt_path, "utt.wav", "selftest")]).compose(0)
        contrasts = []
        for name, stem in scene.stems.items():
            mask = np.zeros(len(stem), bool)
            for s in scene.segments:
                if s["source"] == name:
                    mask[int(s["start"] * scene.sr) : int(s["end"] * scene.sr)] = True
            if mask.any() and (~mask).any():
                contrasts.append(
                    10 * np.log10(np.mean(stem[mask] ** 2) / (np.mean(stem[~mask] ** 2) + 1e-20))
                )
        check(
            "labels sit on the sources",
            bool(contrasts) and min(contrasts) > 10.0,
            f"energy inside vs outside labels: {', '.join(f'{c:.0f} dB' for c in contrasts)}",
        )
    finally:
        utt_path.unlink(missing_ok=True)

    if scratch:
        scratch.mkdir(parents=True, exist_ok=True)
        sf.write(scratch / "source.wav", speech, SR, subtype="PCM_16")
        sf.write(scratch / "live.wav", live.audio, live.sr, subtype="PCM_16")
        sf.write(scratch / "device.wav", device.audio, device.sr, subtype="PCM_16")
        sf.write(scratch / "scene.wav", scene.audio, scene.sr, subtype="PCM_16")
        print(f"\ndemo audio written to {scratch}")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0
