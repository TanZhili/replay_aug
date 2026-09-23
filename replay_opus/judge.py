"""Judge rendered audio with independent replay detectors.

For every pair in a ``render`` output this rebuilds the scene and scores four
conditions with each detector:

``dry``     the input TTS, no room at all
``live``    the rendered live-talker file
``device``  the rendered device-playback file
``replay``  positive control, rebuilt here: the live recording itself, replayed through
            the same device at the same spot in the same scene - the ASVspoof definition
            of a replay attack (two acoustic passes)

One score on its own says little: the public checkpoints were trained on ASVspoof
physical-access data, where the bona fide class is a single loudspeaker pass and the
spoof class a recorded-then-replayed second pass. The comparisons carry the meaning:

* ``replay`` vs ``live`` - can the detector see a replay in our scenes at all? Near 0.5
  means no, and then its view of the device class is uninformative.
* ``live`` vs ``dry`` - does it simply react to room and noise? If so, only same-scene
  comparisons mean anything.
* ``device`` vs ``live`` - how much replay evidence it finds in our device renderings,
  within the same scene; set it against the ``replay`` row. It is also broken down by
  playback form: forms with a recording stage in front of the loudspeaker should read
  as more replayed than a single pass.

``--reference`` scores a list of your own labelled recordings as well, which is the
only way to learn whether a detector works on real audio like yours.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .audit import auc
from .capture import to_digital
from .config import Config
from .detectors import DETECTORS, load_detector, prepare
from .dsp import fade_edges, fit_length, resample_to, to_mono
from .pathlist import read_utterances
from .render import UtteranceRenderer
from .scene import derive_seed

CONDITIONS = ("dry", "live", "device", "replay")
SPOOF_LABELS = {"spoof", "replay", "device", "fake", "1"}
BONAFIDE_LABELS = {"bonafide", "bona_fide", "genuine", "live", "real", "0"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def _load_records(manifest: Path, limit: int | None) -> dict[str, dict[str, dict]]:
    pairs: dict[str, dict[str, dict]] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "error" in record or record.get("label") not in ("live", "device"):
            continue
        pairs.setdefault(record["pair"], {})[record["label"]] = record
        if limit and len(pairs) > limit:
            pairs.pop(record["pair"])
            break
    return pairs


def _replay_control(renderer: UtteranceRenderer, record: dict) -> tuple[np.ndarray, int]:
    """Record the live talker at the scene's microphone, then replay that recording
    through the scene's device - the same device the pair's device file used."""
    config = renderer.config
    sr = config.sr_internal
    scenes = renderer.scenes
    scene = scenes.sample_scene(
        derive_seed(config.seed, record["key"], record["variant"]), record["key"]
    )
    if scene.describe()["room"] != record["scene"]["room"]:
        raise ValueError(
            f"rebuilding {record['pair']} gave a different room: the config is not the one this "
            "output was rendered with"
        )
    audio, sr_in = sf.read(record["input"], always_2d=False)
    dry = fade_edges(resample_to(to_mono(audio), sr_in, sr), sr, 5.0)
    n = len(dry) + int(sr * config.render.tail_ms / 1000.0)
    noise, _ = scenes.ambient(scene, n)
    live = scenes.render_talker(scene, dry)
    recording = to_digital(fit_length(live.signal, n) + noise, sr, scene.capture)
    replay = scenes.render_device(scene, recording, raw_programme=True)
    out, _ = scenes.capture(scene, fit_length(replay.signal, n) + noise)
    return out, config.output.sr


def _summary(scores: dict[str, dict[str, float]], meta: dict[str, dict]) -> dict:
    by_condition = {c: np.array([s[c] for s in scores.values() if c in s]) for c in CONDITIONS}
    out: dict = {
        "median": {},
        "auc": {},
        "paired": {},
        "by_device": {},
        "by_playback_form": {},
        "by_environment": {},
    }
    for condition, values in by_condition.items():
        if len(values):
            out["median"][condition] = round(float(np.median(values)), 3)
    for high, low in (("device", "live"), ("replay", "live"), ("live", "dry"), ("device", "dry")):
        if len(by_condition[high]) and len(by_condition[low]):
            out["auc"][f"{high}>{low}"] = round(auc(by_condition[high], by_condition[low]), 3)
    for high in ("device", "replay"):
        diffs = np.array(
            [s[high] - s["live"] for s in scores.values() if high in s and "live" in s]
        )
        if len(diffs):
            wins = int(np.sum(diffs > 0))
            lo, hi = wilson(wins, len(diffs))
            out["paired"][f"{high}>live"] = {
                "n": len(diffs),
                "win_rate": round(wins / len(diffs), 3),
                "ci95": [round(lo, 3), round(hi, 3)],
                "median_diff": round(float(np.median(diffs)), 3),
            }
    for field, key in (
        ("by_device", "device"),
        ("by_playback_form", "playback_form"),
        ("by_environment", "environment"),
    ):
        groups: dict[str, list[float]] = defaultdict(list)
        for pair, s in scores.items():
            if "device" in s and "live" in s and meta[pair].get(key) is not None:
                groups[meta[pair][key]].append(s["device"] - s["live"])
        for name, diffs in sorted(groups.items()):
            wins = int(np.sum(np.array(diffs) > 0))
            out[field][name] = {
                "n": len(diffs),
                "win_rate": round(wins / len(diffs), 3),
                "median_diff": round(float(np.median(diffs)), 3),
            }
    return out


def _reference_scores(detector, reference: Path, level_dbfs: float | None) -> dict:
    items = read_utterances(reference)
    positives, negatives = [], []
    for item in items:
        label = (item.text or "").strip().lower()
        if label not in SPOOF_LABELS | BONAFIDE_LABELS:
            raise ValueError(
                f"{reference}: label {item.text!r} for {item.key} is not bonafide/spoof"
            )
        audio, sr = sf.read(str(item.path), always_2d=False)
        score = detector.score(prepare(to_mono(audio), sr, level_dbfs))
        (positives if label in SPOOF_LABELS else negatives).append(score)
    return {
        "n_spoof": len(positives),
        "n_bonafide": len(negatives),
        "auc_spoof>bonafide": round(auc(np.array(positives), np.array(negatives)), 3),
    }


def run_judge(
    manifest: str | Path,
    detectors: list[str] | None = None,
    limit: int | None = None,
    controls: bool = True,
    reference: str | Path | None = None,
    config_path: str | Path | None = None,
    level_dbfs: float | None = -26.0,
    progress=None,
) -> dict:
    manifest = Path(manifest)
    root = manifest.parent
    pairs = _load_records(manifest, limit)
    if not pairs:
        raise ValueError(f"{manifest} holds no rendered live/device clips")

    saved = Path(config_path) if config_path else root / "config.yaml"
    renderer = None
    if controls:
        if not saved.is_file():
            raise ValueError(
                f"{saved} not found: the replay control rebuilds each scene from the config used "
                "to render. Pass --config, or --no-controls to skip the controls."
            )
        renderer = UtteranceRenderer(
            Config.from_dict(yaml.safe_load(saved.read_text(encoding="utf-8")))
        )

    clips: dict[str, dict[str, np.ndarray]] = {}
    meta: dict[str, dict] = {}
    for done, (pair, records) in enumerate(pairs.items(), start=1):
        any_record = next(iter(records.values()))
        device_record = records.get("device")
        meta[pair] = {
            "environment": any_record["scene"]["environment"],
            "device": device_record["source"]["device"] if device_record else None,
            "playback_form": device_record.get("playback_form") if device_record else None,
        }
        clips[pair] = {}
        for label, record in records.items():
            audio, sr = sf.read(root / record["audio"], always_2d=False)
            clips[pair][label] = prepare(audio, sr, level_dbfs)
        if controls:
            audio, sr = sf.read(any_record["input"], always_2d=False)
            clips[pair]["dry"] = prepare(to_mono(audio), sr, level_dbfs)
            clips[pair]["replay"] = prepare(*_replay_control(renderer, any_record), level_dbfs)
        if progress:
            progress("prepare", done, len(pairs))

    result: dict = {
        "manifest": str(manifest),
        "n_pairs": len(pairs),
        "level_dbfs": level_dbfs,
        "detectors": {},
    }
    for name in detectors or list(DETECTORS):
        detector = load_detector(name)
        scores: dict[str, dict[str, float]] = {}
        for done, (pair, conditions) in enumerate(clips.items(), start=1):
            scores[pair] = {c: round(detector.score(x), 4) for c, x in conditions.items()}
            if progress:
                progress(name, done, len(clips))
        entry = {"title": detector.title, **_summary(scores, meta), "scores": scores}
        if reference:
            entry["reference"] = _reference_scores(detector, Path(reference), level_dbfs)
        result["detectors"][name] = entry
    return result


def format_judge_report(result: dict) -> str:
    lines = [f"{result['n_pairs']} scenes from {result['manifest']}"]
    for name, d in result["detectors"].items():
        lines += ["", f"== {name}: {d['title']}", "   replay score, higher = more replayed"]
        lines.append("   median   " + "  ".join(f"{c} {v:+.2f}" for c, v in d["median"].items()))
        lines.append("   AUC      " + "  ".join(f"{k} {v:.3f}" for k, v in d["auc"].items()))
        for key, p in d["paired"].items():
            lo, hi = p["ci95"]
            lines.append(
                f"   same scene, {key}: {p['win_rate']:.0%} of {p['n']} "
                f"(95% CI {lo:.0%}-{hi:.0%}), median difference {p['median_diff']:+.2f}"
            )
        for field, title in (
            ("by_device", "device"),
            ("by_playback_form", "playback form"),
            ("by_environment", "environment"),
        ):
            if d.get(field):
                cells = [f"{k} {v['win_rate']:.0%} (n={v['n']})" for k, v in d[field].items()]
                lines.append(f"   device>live by {title}: " + ", ".join(cells))
        if "reference" in d:
            r = d["reference"]
            lines.append(
                f"   your reference set: AUC spoof>bonafide {r['auc_spoof>bonafide']:.3f} "
                f"({r['n_spoof']} spoof, {r['n_bonafide']} bona fide)"
            )
    lines += [
        "",
        "How to read this:",
        "  replay>live near 1.0 : the detector sees an ASVspoof-style replay in these scenes;",
        "                         near 0.5 it cannot, and its verdict on 'device' says nothing.",
        "  live>dry well above 0.5 : it reacts to room and noise alone, so only same-scene",
        "                            comparisons count.",
        "  device>live : replay evidence in the device renderings; compare with replay>live.",
        "  These checkpoints learned the second recording pass of ASVspoof replays, not mouth vs",
        "  loudspeaker; a real reference set (--reference) is what shows whether one transfers.",
    ]
    return "\n".join(lines)


__all__ = ["CONDITIONS", "format_judge_report", "run_judge", "wilson"]
