"""Command line interface: ``python -m replay_opus <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from .chain import ReplayChain, describe_config
from .config import Config
from .irbank import IRBank, filter_ir_stats
from .measure import SweepSpec, measure_chain, sweep_with_padding
from .metrics import describe_ir
from .selftest import run_selftest

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus")

_WORKER: dict[str, object] = {}


def find_audio(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


def _init_worker(config_dict: dict) -> None:
    _WORKER["chain"] = ReplayChain(Config.from_dict(config_dict))


def _process_one(args: tuple[str, str, str, int]) -> dict:
    in_path, out_path, key, variant = args
    chain: ReplayChain = _WORKER["chain"]  # type: ignore[assignment]
    audio, sr = sf.read(in_path, always_2d=False)
    result = chain.process(np.asarray(audio, dtype=np.float64), sr, key, variant)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, result.audio, result.sr, subtype=chain.config.output.subtype)
    return {"input": in_path, "output": out_path, **result.meta}


def cmd_simulate(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.variants is not None:
        config.variants_per_file = args.variants
    if args.room_ir_dir:
        config.room.ir_dir = args.room_ir_dir
        config.room.mode = args.room_mode or "reshaped"
    elif args.room_mode:
        config.room.mode = args.room_mode

    in_root, out_root = Path(args.input), Path(args.output)
    files = find_audio(in_root)
    if not files:
        print(f"no audio found under {in_root}", file=sys.stderr)
        return 1

    probe = ReplayChain(config)
    print(json.dumps(describe_config(probe), ensure_ascii=False))
    print(f"{len(files)} input files x {config.variants_per_file} variants")

    jobs = []
    for path in files:
        key = str(path.relative_to(in_root)).replace(os.sep, "/")
        for variant in range(config.variants_per_file):
            stem = path.relative_to(in_root).with_suffix("")
            name = (
                f"{stem}{config.output.suffix}"
                if config.variants_per_file == 1
                else (f"{stem}__v{variant}{config.output.suffix}")
            )
            jobs.append((str(path), str(out_root / name), key, variant))

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    written = failed = 0

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        if args.workers == 1:
            _init_worker(config.to_dict())
            iterator = (_safe(_process_one, job) for job in jobs)
            for record in iterator:
                written, failed = _record(manifest, record, written, failed, len(jobs))
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker, initargs=(config.to_dict(),)
            ) as pool:
                futures = {pool.submit(_process_one, job): job for job in jobs}
                for future in as_completed(futures):
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                        record = {"input": futures[future][0], "error": repr(exc)}
                    written, failed = _record(manifest, record, written, failed, len(jobs))

    print(f"\nwrote {written} files, {failed} failed")
    print(f"manifest: {manifest_path}")
    return 1 if failed and not written else 0


def _safe(func, job):
    try:
        return func(job)
    except Exception as exc:  # noqa: BLE001
        return {"input": job[0], "error": repr(exc)}


def _record(manifest, record: dict, written: int, failed: int, total: int) -> tuple[int, int]:
    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    if "error" in record:
        failed += 1
        print(f"  FAILED {record['input']}: {record['error']}", file=sys.stderr)
    else:
        written += 1
    done = written + failed
    if done % 25 == 0 or done == total:
        print(f"  {done}/{total}", end="\r", flush=True)
    return written, failed


def cmd_sweep(args: argparse.Namespace) -> int:
    spec = SweepSpec(f1=args.f1, f2=args.f2, duration=args.duration, sr=args.sr)
    padded, _, rate, pad = sweep_with_padding(spec)
    sf.write(args.output, padded, spec.sr, subtype="PCM_24")
    print(f"wrote {args.output}  ({len(padded) / spec.sr:.1f}s @ {spec.sr} Hz, L={rate:.4f}s)")
    print(
        "Play this file through the loudspeaker and record it with the microphone you "
        "want to model, keeping both at a fixed gain. Then run 'measure extract'."
    )
    print(f"Leading silence is {pad / spec.sr:.1f}s; keep it in the recording.")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    recorded, sr = sf.read(args.recording, always_2d=False)
    recorded = np.asarray(recorded, dtype=np.float64)
    if recorded.ndim > 1:
        recorded = recorded.mean(axis=1)

    spec = SweepSpec(f1=args.f1, f2=args.f2, duration=args.duration, sr=sr)
    ir, model, info = measure_chain(recorded, spec, ir_ms=args.ir_ms)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.recording).stem
    sf.write(out / f"{name}.wav", ir / (np.max(np.abs(ir)) + 1e-12) * 0.95, sr, subtype="PCM_24")
    model.label = name
    model.save(out / f"{name}.npz")

    report = {"name": name, "sr": sr, **describe_ir(ir, sr), **info}
    (out / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "harmonic_profile"}, indent=2))
    print(f"harmonic levels (dB re fundamental): {info['harmonic_profile']['level_db']}")
    print(f"\nwrote {out / name}.wav (room IR), .npz (speaker model), .json (report)")
    return 0


def cmd_ir_stats(args: argparse.Namespace) -> int:
    bank = IRBank.from_dir(args.ir_dir, args.sr)
    rows = bank.stats()
    kept = filter_ir_stats(rows)
    print(f"{len(rows)} impulse responses, {len(kept)} within usable T60/DRR bounds")
    if kept:
        for field_name in ("t60_s", "drr_db", "c50_db"):
            values = np.array([r[field_name] for r in kept])
            print(
                f"  {field_name:8s} min={values.min():7.3f} "
                f"median={np.median(values):7.3f} max={values.max():7.3f}"
            )
    if args.output:
        Path(args.output).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )
        print(f"wrote {args.output}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    return run_selftest(Path(args.output) if args.output else None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay_opus",
        description="Batch-convert clean/TTS audio into played-and-re-recorded audio.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run the replay chain over a directory")
    sim.add_argument("--input", required=True, help="directory of source audio")
    sim.add_argument("--output", required=True, help="directory for generated audio")
    sim.add_argument("--config", help="YAML config; defaults are used when omitted")
    sim.add_argument("--seed", type=int, help="override config seed")
    sim.add_argument("--variants", type=int, help="replay conditions per input file")
    sim.add_argument("--room-ir-dir", help="directory of room IRs; implies room mode 'reshaped'")
    sim.add_argument("--room-mode", choices=["measured", "reshaped", "synthetic", "simulated"])
    sim.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    sim.set_defaults(func=cmd_simulate)

    sweep = sub.add_parser("sweep", help="write the measurement sweep to play back")
    sweep.add_argument("--output", default="sweep.wav")
    sweep.add_argument("--f1", type=float, default=50.0)
    sweep.add_argument("--f2", type=float, default=20000.0)
    sweep.add_argument("--duration", type=float, default=12.0)
    sweep.add_argument("--sr", type=int, default=48000)
    sweep.set_defaults(func=cmd_sweep)

    extract = sub.add_parser("extract", help="turn a recorded sweep into an IR + speaker model")
    extract.add_argument("--recording", required=True, help="the recorded sweep")
    extract.add_argument("--output", required=True, help="output directory")
    extract.add_argument("--name", help="basename for the generated files")
    extract.add_argument("--f1", type=float, default=50.0)
    extract.add_argument("--f2", type=float, default=20000.0)
    extract.add_argument("--duration", type=float, default=12.0)
    extract.add_argument("--ir-ms", type=float, default=1500.0)
    extract.set_defaults(func=cmd_extract)

    stats = sub.add_parser("ir-stats", help="report T60 / DRR / C50 over an IR directory")
    stats.add_argument("--ir-dir", required=True)
    stats.add_argument("--sr", type=int, default=48000)
    stats.add_argument("--output", help="write per-IR records as JSONL")
    stats.set_defaults(func=cmd_ir_stats)

    test = sub.add_parser("selftest", help="verify the installation against known systems")
    test.add_argument("--output", help="directory for demo audio")
    test.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
