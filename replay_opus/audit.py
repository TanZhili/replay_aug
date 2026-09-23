"""Shortcut audit: which trivial features separate the live and device classes?

A rejection model learns whatever separates its training classes most cheaply. If
the device class is simply quieter, or shorter, or noisier, the model will learn
that instead of anything about loudspeakers - and fail on a loud TV or a quiet user.
This computes a handful of per-file features on a ``render`` manifest and reports,
for each, the AUC of that feature alone at telling device from live (0.5 = no
information, 1.0 or 0.0 = perfect separation).

Two kinds of feature are reported separately. *Nuisance* features (level, peak,
duration, noise floor) should stay near 0.5; strong separation there is a shortcut
to fix in the configuration. *Acoustic* features (low/high band balance, spectral
centroid and flatness, crest factor) are where loudspeaker physics shows up, so they
are expected to carry information - that is the point of the data.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.stats import rankdata

from .dsp import EPS

NUISANCE = ("level_dbfs", "peak_dbfs", "duration_s", "noise_floor_dbfs")
ACOUSTIC = ("lf_ratio_db", "hf_ratio_db", "centroid_hz", "flatness", "crest_db", "dynamic_range_db")
SHORTCUT_MARGIN = 0.2


def features(x: np.ndarray, sr: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    hop = max(1, int(0.02 * sr))
    n_frames = max(1, len(x) // hop)
    frames = x[: n_frames * hop].reshape(n_frames, hop) if len(x) >= hop else x[None, :]
    energy_db = 10.0 * np.log10(np.mean(frames**2, axis=1) + EPS)
    active = energy_db > energy_db.max() - 30.0

    freqs, _, spec = signal.stft(x, fs=sr, nperseg=512, noverlap=256)
    power = np.abs(spec) ** 2
    frame_db = 10.0 * np.log10(power.sum(axis=0) + EPS)
    loud = frame_db > frame_db.max() - 30.0
    p = power[:, loud] if np.any(loud) else power
    band = p.sum(axis=1)
    total = band.sum() + EPS
    active_level = 10.0 * np.log10(np.mean(10.0 ** (energy_db[active] / 10.0)) + EPS)
    return {
        "level_dbfs": float(active_level),
        "peak_dbfs": float(20.0 * np.log10(np.max(np.abs(x)) + EPS)),
        "duration_s": float(len(x) / sr),
        "noise_floor_dbfs": float(np.percentile(energy_db, 10)),
        "dynamic_range_db": float(np.percentile(energy_db, 95) - np.percentile(energy_db, 10)),
        "lf_ratio_db": float(10.0 * np.log10(band[freqs < 250.0].sum() / total + EPS)),
        "hf_ratio_db": float(
            10.0 * np.log10(band[freqs > min(4000.0, 0.4 * sr)].sum() / total + EPS)
        ),
        "centroid_hz": float(np.sum(freqs * band) / total),
        "flatness": float(
            np.mean(np.exp(np.mean(np.log(p + EPS), axis=0)) / (np.mean(p, axis=0) + EPS))
        ),
        "crest_db": float(20.0 * np.log10(np.max(np.abs(x)) + EPS) - active_level),
    }


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(feature of a random positive > feature of a random negative), ties halved."""
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([positive, negative]))
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def run_audit(manifest: str | Path, limit: int | None = None) -> dict:
    manifest = Path(manifest)
    root = manifest.parent
    rows = []
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "error" in record or record.get("label") not in ("live", "device"):
                continue
            rows.append(record)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"{manifest} holds no rendered live/device clips")

    table: dict[str, dict[str, list[float]]] = {
        "live": defaultdict(list),
        "device": defaultdict(list),
    }
    by_env: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"live": [], "device": []})
    pairs: dict[str, dict[str, float]] = defaultdict(dict)
    for record in rows:
        audio, sr = sf.read(str(root / record["audio"]), always_2d=False)
        feats = features(audio, sr)
        label = record["label"]
        for name, value in feats.items():
            table[label][name].append(value)
        by_env[record["scene"]["environment"]][label].append(feats["level_dbfs"])
        pairs[f"{record['key']}#{record['variant']}"][label] = feats["level_dbfs"]

    result: dict = {
        "n_live": len(table["live"]["level_dbfs"]),
        "n_device": len(table["device"]["level_dbfs"]),
        "features": {},
        "level_auc_by_environment": {},
    }
    for name in (*NUISANCE, *ACOUSTIC):
        live = np.asarray(table["live"][name])
        device = np.asarray(table["device"][name])
        score = auc(device, live)
        result["features"][name] = {
            "kind": "nuisance" if name in NUISANCE else "acoustic",
            "auc_device_gt_live": round(score, 3),
            "live_median": round(float(np.median(live)), 3) if len(live) else None,
            "device_median": round(float(np.median(device)), 3) if len(device) else None,
            "shortcut": bool(
                name in NUISANCE and np.isfinite(score) and abs(score - 0.5) > SHORTCUT_MARGIN
            ),
        }
    for env, values in sorted(by_env.items()):
        result["level_auc_by_environment"][env] = round(
            auc(np.asarray(values["device"]), np.asarray(values["live"])), 3
        )
    diffs = [p["device"] - p["live"] for p in pairs.values() if "device" in p and "live" in p]
    if diffs:
        result["paired_level_difference_db"] = {
            "n": len(diffs),
            "mean": round(float(np.mean(diffs)), 2),
            "std": round(float(np.std(diffs)), 2),
            "p10": round(float(np.percentile(diffs, 10)), 2),
            "p90": round(float(np.percentile(diffs, 90)), 2),
        }
    return result


def format_report(result: dict) -> str:
    lines = [f"{result['n_live']} live clips, {result['n_device']} device clips", ""]
    lines.append(f"{'feature':<20}{'kind':<10}{'AUC':>7}{'live':>12}{'device':>12}")
    lines.append("-" * 61)
    for name, row in result["features"].items():
        flag = "  <- possible shortcut" if row["shortcut"] else ""
        lines.append(
            f"{name:<20}{row['kind']:<10}{row['auc_device_gt_live']:>7.3f}"
            f"{row['live_median']:>12.3f}{row['device_median']:>12.3f}{flag}"
        )
    lines.append("")
    lines.append(
        "level AUC by environment: "
        + ", ".join(
            f"{env} {score:.2f}" for env, score in result["level_auc_by_environment"].items()
        )
    )
    if "paired_level_difference_db" in result:
        d = result["paired_level_difference_db"]
        lines.append(
            f"paired level difference device - live: mean {d['mean']:+.1f} dB, sd {d['std']:.1f}, "
            f"10-90% [{d['p10']:+.1f}, {d['p90']:+.1f}] over {d['n']} pairs"
        )
    lines.append("")
    lines.append(
        f"Nuisance features should stay within 0.5 +/- {SHORTCUT_MARGIN}; acoustic features are "
        "expected to separate the classes."
    )
    return "\n".join(lines)


__all__ = ["auc", "features", "format_report", "run_audit"]
