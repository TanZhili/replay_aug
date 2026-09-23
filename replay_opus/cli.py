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
import yaml

from .config import Config
from .detectors import DETECTORS
from .irbank import IRBank, filter_ir_stats
from .measure import SweepSpec, measure_chain, sweep_with_padding
from .metrics import describe_ir
from .pathlist import Utterance, read_utterances

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus")

_WORKER: dict[str, object] = {}


# ---------------------------------------------------------------------------
# config handling
# ---------------------------------------------------------------------------


def _apply_set(data: dict, assignment: str) -> None:
    """``a.b.c=value`` with the value parsed as YAML (numbers, lists, null...)."""
    if "=" not in assignment:
        raise ValueError(f"--set expects key=value, got {assignment!r}")
    key, _, raw = assignment.partition("=")
    parts = key.strip().split(".")
    node = data
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {} if node.get(part) is None else node[part]
        node = node[part]
    node[parts[-1]] = yaml.safe_load(raw)


def save_config(config: Config, out_root: Path) -> None:
    """Keep the effective config with the outputs: scenes can be rebuilt from it."""
    text = yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False)
    (out_root / "config.yaml").write_text(text, encoding="utf-8")


def load_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config)
    if getattr(args, "set", None):
        data = config.to_dict()
        for assignment in args.set:
            _apply_set(data, assignment)
        config = Config.from_dict(data)
    if getattr(args, "seed", None) is not None:
        config.seed = args.seed
    if getattr(args, "variants", None) is not None:
        config.variants_per_file = args.variants
    if getattr(args, "mode", None):
        config.render.mode = args.mode
    config.validate()
    return config


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------


def _init_render_worker(config_dict: dict) -> None:
    from .render import UtteranceRenderer  # noqa: PLC0415 - keep CLI start-up light

    _WORKER["renderer"] = UtteranceRenderer(Config.from_dict(config_dict))


def _render_job(job: tuple[str, str, str | None, str | None, int, str, int]) -> list[dict]:
    path, key, text, speaker, variant, out_root, n_variants = job
    renderer = _WORKER["renderer"]
    audio, sr = sf.read(path, always_2d=False)
    clips = renderer.render(np.asarray(audio, dtype=np.float64), sr, key, variant)
    stem = Path(key).with_suffix("").as_posix()
    output = renderer.config.output
    records = []
    for clip in clips:
        name = stem if n_variants == 1 else f"{stem}__v{variant}"
        rel = f"{clip.label}/{name}{output.suffix}"
        target = Path(out_root) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, clip.audio, clip.sr, subtype=output.subtype)
        records.append(
            {
                "audio": rel,
                "input": path,
                "pair": f"{key}#{variant}",
                "text": text,
                "speaker": speaker,
                **clip.meta,
            }
        )
    return records


def _init_scene_worker(
    config_dict: dict, utterances: list[tuple[str, str, str | None, str | None]]
) -> None:
    from .compose import SceneComposer  # noqa: PLC0415

    utts = [Utterance(Path(p), k, t, s) for p, k, t, s in utterances]
    _WORKER["composer"] = SceneComposer(Config.from_dict(config_dict), utts)


def _scene_job(job: tuple[int, str]) -> list[dict]:
    index, out_root = job
    composer = _WORKER["composer"]
    scene = composer.compose(index)
    name = f"scene_{index:06d}"
    output = composer.config.output
    subtype = output.subtype
    rel = f"scenes/{name}{output.suffix}"
    target = Path(out_root) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, scene.audio, scene.sr, subtype=subtype)
    record = {"audio": rel, "scene_id": name, **scene.meta}
    if scene.stems:
        record["stems"] = {}
        for stem_name, stem in scene.stems.items():
            stem_rel = f"stems/{name}.{stem_name}{output.suffix}"
            (Path(out_root) / stem_rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(Path(out_root) / stem_rel, np.clip(stem, -1.0, 1.0), scene.sr, subtype=subtype)
            record["stems"][stem_name] = stem_rel
    return [record]


def _run_jobs(
    jobs, worker, initializer, initargs, workers: int, manifest_path: Path
) -> tuple[int, int]:
    written = failed = 0
    total = len(jobs)

    def record_all(records: list[dict], manifest) -> None:
        nonlocal written, failed
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            if "error" in record:
                failed += 1
                print(
                    f"  FAILED {record.get('input', record.get('job'))}: {record['error']}",
                    file=sys.stderr,
                )
            else:
                written += 1

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        if workers == 1:
            initializer(*initargs)
            for done, job in enumerate(jobs, start=1):
                try:
                    records = worker(job)
                except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                    records = [{"job": repr(job[:2]), "error": repr(exc)}]
                record_all(records, manifest)
                if done % 10 == 0 or done == total:
                    print(f"  {done}/{total}", end="\r", flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=initializer, initargs=initargs
            ) as pool:
                futures = {pool.submit(worker, job): job for job in jobs}
                for done, future in enumerate(as_completed(futures), start=1):
                    try:
                        records = future.result()
                    except Exception as exc:  # noqa: BLE001
                        records = [{"job": repr(futures[future][:2]), "error": repr(exc)}]
                    record_all(records, manifest)
                    if done % 10 == 0 or done == total:
                        print(f"  {done}/{total}", end="\r", flush=True)
    print()
    return written, failed


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_render(args: argparse.Namespace) -> int:
    try:
        config = load_config(args)
        utterances = read_utterances(args.input, suffixes=AUDIO_SUFFIXES)
        _init_render_worker(config.to_dict())  # fail fast on bad lists before forking
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    _WORKER.clear()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    save_config(config, out_root)
    n_variants = config.variants_per_file
    jobs = [
        (str(u.path), u.key, u.text, u.speaker, v, str(out_root), n_variants)
        for u in utterances
        for v in range(n_variants)
    ]
    per_job = 2 if config.render.mode == "paired" else 1
    print(
        f"{len(utterances)} utterances x {n_variants} variants, mode {config.render.mode} "
        f"-> {len(jobs) * per_job} files"
    )
    manifest = out_root / "manifest.jsonl"
    written, failed = _run_jobs(
        jobs, _render_job, _init_render_worker, (config.to_dict(),), args.workers, manifest
    )
    print(f"wrote {written} files, {failed} failed\nmanifest: {manifest}")
    return 1 if failed and not written else 0


def cmd_mix(args: argparse.Namespace) -> int:
    from .mix import main as mix_main  # noqa: PLC0415

    try:
        config = load_config(args)
        return mix_main(
            args.speech_list,
            args.playback_list,
            args.noise_list,
            args.rir_list,
            args.output,
            config=config,
            count=args.count,
            workers=args.workers,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1


def cmd_scenes(args: argparse.Namespace) -> int:
    try:
        config = load_config(args)
        if args.stems:
            config.scene.save_stems = True
        utterances = read_utterances(args.input, suffixes=AUDIO_SUFFIXES)
        packed = [(str(u.path), u.key, u.text, u.speaker) for u in utterances]
        _init_scene_worker(config.to_dict(), packed)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    _WORKER.clear()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    save_config(config, out_root)
    jobs = [(i, str(out_root)) for i in range(args.start, args.start + args.count)]
    print(f"{args.count} scenes from {len(utterances)} utterances")
    manifest = out_root / "manifest.jsonl"
    written, failed = _run_jobs(
        jobs, _scene_job, _init_scene_worker, (config.to_dict(), packed), args.workers, manifest
    )
    print(f"wrote {written} scenes, {failed} failed\nmanifest: {manifest}")
    return 1 if failed and not written else 0


def cmd_audit(args: argparse.Namespace) -> int:
    from .audit import format_report, run_audit  # noqa: PLC0415

    try:
        result = run_audit(args.manifest, args.limit)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(format_report(result))
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.output}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    from .judge import format_judge_report, run_judge  # noqa: PLC0415 - pulls in torch

    def progress(stage: str, done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {stage} {done}/{total}      ", end="\r", flush=True)

    try:
        result = run_judge(
            args.manifest,
            args.detector,
            limit=args.limit,
            controls=not args.no_controls,
            reference=args.reference,
            config_path=args.config,
            level_dbfs=args.level,
            progress=progress,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print()
    print(format_judge_report(result))
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.output}")
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    config = load_config(args)
    tables = {
        "environments": config.environment_table(),
        "devices": config.device_table(),
        "content": config.content_table(),
    }
    wanted = {args.section: tables[args.section]} if args.section else tables
    text = yaml.safe_dump(
        wanted, allow_unicode=True, sort_keys=False, default_flow_style=None, width=100
    )
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    spec = SweepSpec(f1=args.f1, f2=args.f2, duration=args.duration, sr=args.sr)
    padded, _, rate, pad = sweep_with_padding(spec)
    sf.write(args.output, padded, spec.sr, subtype="PCM_24")
    print(f"wrote {args.output}  ({len(padded) / spec.sr:.1f}s @ {spec.sr} Hz, L={rate:.4f}s)")
    print(
        "Play it through the device you want to model and record it 10-30 cm in front of "
        "the device with a flat microphone, at a fixed gain. Then run 'extract'."
    )
    print(f"Leading silence is {pad / spec.sr:.1f}s; keep it in the recording.")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    recorded, sr = sf.read(args.recording, always_2d=False)
    recorded = np.asarray(recorded, dtype=np.float64)
    if recorded.ndim > 1:
        recorded = recorded.mean(axis=1)

    spec = SweepSpec(f1=args.f1, f2=args.f2, duration=args.duration, sr=sr)
    ir, info = measure_chain(recorded, spec, ir_ms=args.ir_ms)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.recording).stem
    sf.write(out / f"{name}.wav", ir / (np.max(np.abs(ir)) + 1e-12) * 0.95, sr, subtype="PCM_24")
    base = ir[: min(len(ir), 4096)]
    np.savez(
        out / f"{name}.npz",
        g1=base / (np.max(np.abs(base)) + 1e-12),
        sr=sr,
        label=name,
        harmonic_rolloff_db=info["harmonic_rolloff_db"],
        thd=info["thd_at_sweep_level"],
    )

    report = {"name": name, "sr": sr, **describe_ir(ir, sr), **info}
    (out / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "harmonic_profile"}, indent=2))
    print(f"harmonic levels (dB re fundamental): {info['harmonic_profile']['level_db']}")
    print(
        f"\nwrote {out / name}.wav (impulse response), .npz (for devices.measured), .json (report)"
    )
    return 0


def cmd_ir_stats(args: argparse.Namespace) -> int:
    try:
        bank = IRBank.from_list(args.ir_list, args.sr)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
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
    from .selftest import run_selftest  # noqa: PLC0415

    return run_selftest(Path(args.output) if args.output else None)


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML config; built-in defaults when omitted")
    parser.add_argument("--seed", type=int, help="override the config seed")
    parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override any config value, e.g. --set capture.agc_prob=0 (repeatable)",
    )


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - one flat list of options
    parser = argparse.ArgumentParser(
        prog="replay_opus",
        description="Render TTS speech as live talkers and as device playback in shared scenes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    default_workers = max(1, (os.cpu_count() or 2) // 2)

    render = sub.add_parser("render", help="per-utterance live / device clips (paired by default)")
    render.add_argument(
        "--input", required=True, help="utterance list: one path per line, optional text"
    )
    render.add_argument("--output", required=True, help="output directory")
    render.add_argument("--variants", type=int, help="scenes per utterance")
    render.add_argument("--mode", choices=["paired", "mixed", "live", "device"])
    render.add_argument("--workers", type=int, default=default_workers)
    _add_config_args(render)
    render.set_defaults(func=cmd_render)

    scenes = sub.add_parser(
        "scenes", help="full-duplex scenes with time-aligned live / device labels"
    )
    scenes.add_argument(
        "--input", required=True, help="utterance list shared by users and programmes"
    )
    scenes.add_argument("--output", required=True, help="output directory")
    scenes.add_argument("--count", type=int, required=True, help="number of scenes")
    scenes.add_argument("--start", type=int, default=0, help="first scene index (for sharding)")
    scenes.add_argument("--stems", action="store_true", help="also write per-source stems")
    scenes.add_argument("--workers", type=int, default=default_workers)
    _add_config_args(scenes)
    scenes.set_defaults(func=cmd_scenes)

    mix = sub.add_parser(
        "mix",
        help="microphone recording of user speech and device playback in one environment",
    )
    mix.add_argument("--speech-list", help="user speech (A); one path per line, optional text")
    mix.add_argument(
        "--playback-list",
        help="audio rendered as device playback (B); may be the same list as --speech-list",
    )
    mix.add_argument("--noise-list", help="noise recordings, one path per line")
    mix.add_argument("--rir-list", help="measured room impulse responses, one path per line")
    mix.add_argument("--output", required=True, help="output directory")
    mix.add_argument(
        "--count",
        type=int,
        help="number of mixes; default is the longest list",
    )
    mix.add_argument("--workers", type=int, default=default_workers)
    _add_config_args(mix)
    mix.set_defaults(func=cmd_mix)

    audit = sub.add_parser("audit", help="check a render manifest for level / duration shortcuts")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--limit", type=int, help="audit only the first N clips")
    audit.add_argument("--output", help="write the full result as JSON")
    audit.set_defaults(func=cmd_audit)

    judge = sub.add_parser("judge", help="score a render output with third-party replay detectors")
    judge.add_argument("--manifest", required=True, help="manifest.jsonl written by render")
    judge.add_argument(
        "--detector",
        action="append",
        choices=sorted(DETECTORS),
        help="detector to run (repeatable); all of them by default",
    )
    judge.add_argument("--limit", type=int, help="judge only the first N pairs")
    judge.add_argument(
        "--no-controls", action="store_true", help="skip the dry and replay controls"
    )
    judge.add_argument("--reference", help="your own recordings: path<TAB>bonafide|spoof per line")
    judge.add_argument("--config", help="config used to render; default <render dir>/config.yaml")
    judge.add_argument(
        "--level",
        type=float,
        default=-26.0,
        help="active level (dBFS) clips are set to before scoring",
    )
    judge.add_argument("--output", help="write per-clip scores and the summary as JSON")
    judge.set_defaults(func=cmd_judge)

    presets = sub.add_parser(
        "presets", help="print the effective environment / device / content presets"
    )
    presets.add_argument("--section", choices=["environments", "devices", "content"])
    presets.add_argument("--output", help="write YAML to a file instead of stdout")
    _add_config_args(presets)
    presets.set_defaults(func=cmd_presets)

    sweep = sub.add_parser("sweep", help="write the measurement sweep to play through a device")
    sweep.add_argument("--output", default="sweep.wav")
    sweep.add_argument("--f1", type=float, default=50.0)
    sweep.add_argument("--f2", type=float, default=20000.0)
    sweep.add_argument("--duration", type=float, default=12.0)
    sweep.add_argument("--sr", type=int, default=48000)
    sweep.set_defaults(func=cmd_sweep)

    extract = sub.add_parser("extract", help="turn a recorded sweep into a device response")
    extract.add_argument("--recording", required=True, help="the recorded sweep")
    extract.add_argument("--output", required=True, help="output directory")
    extract.add_argument("--name", help="basename for the generated files")
    extract.add_argument("--f1", type=float, default=50.0)
    extract.add_argument("--f2", type=float, default=20000.0)
    extract.add_argument("--duration", type=float, default=12.0)
    extract.add_argument("--ir-ms", type=float, default=1500.0)
    extract.set_defaults(func=cmd_extract)

    stats = sub.add_parser("ir-stats", help="report T60 / DRR / C50 for IRs named in a path list")
    stats.add_argument("--ir-list", required=True, help="text file listing impulse responses")
    stats.add_argument("--sr", type=int, default=48000)
    stats.add_argument("--output", help="write per-IR records as JSONL")
    stats.set_defaults(func=cmd_ir_stats)

    test = sub.add_parser("selftest", help="verify the installation against known answers")
    test.add_argument("--output", help="directory for demo audio")
    test.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
