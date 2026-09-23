"""Shortcut audit: which trivial features separate the live and device classes?

A rejection model learns whatever separates its training classes most cheaply. If
the device class is simply quieter, or shorter, or noisier, or starts its speech
later in the file, the model will learn that instead of anything about loudspeakers
- and fail on a loud TV or a quiet user. This computes a handful of per-file
features on a ``render`` manifest and reports, for each, the AUC of that feature
alone at telling device from live (0.5 = no information, 1.0 or 0.0 = perfect).

Two kinds of feature are reported separately. *Nuisance* features (level, peak,
duration, noise floor, speech onset) should stay near 0.5; strong separation there
is a shortcut to fix in the configuration. *Acoustic* features (low/high band
balance, spectral centroid and flatness, crest factor) are where loudspeaker physics
shows up, so they are expected to carry information - that is the point of the data.

The level check is repeated per playback form and per input origin, and the
manifest's acoustic spectral spread (``sstd``) is summarised per class and form: one
room scatters a transfer function by about 5.6 dB, two rooms in series by about 8.
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

NUISANCE = ("level_dbfs", "peak_dbfs", "duration_s", "noise_floor_dbfs", "lead_s")
ACOUSTIC = ("lf_ratio_db", "hf_ratio_db", "centroid_hz", "flatness", "crest_db", "dynamic_range_db")
SHORTCUT_MARGIN = 0.2


def _lead_s(energy_db: np.ndarray, hop_s: float, rise_db: float = 12.0) -> float:
    """Time until the frame energy first rises ``rise_db`` above the opening frames."""
    floor = float(np.median(energy_db[: max(4, int(0.04 / hop_s))]))
    above = np.flatnonzero(energy_db > floor + rise_db)
    return float(above[0] * hop_s) if above.size else float(len(energy_db) * hop_s)


def features(x: np.ndarray, sr: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    hop = max(1, int(0.02 * sr))
    n_frames = max(1, len(x) // hop)
    frames = x[: n_frames * hop].reshape(n_frames, hop) if len(x) >= hop else x[None, :]
    energy_db = 10.0 * np.log10(np.mean(frames**2, axis=1) + EPS)
    active = energy_db > energy_db.max() - 30.0
    fine_hop = max(1, int(0.005 * sr))
    n_fine = max(1, len(x) // fine_hop)
    fine_db = 10.0 * np.log10(
        np.mean(x[: n_fine * fine_hop].reshape(n_fine, fine_hop) ** 2, axis=1) + EPS
    )

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
        "lead_s": _lead_s(fine_db, fine_hop / sr),
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


def _acoustic_sstd(record: dict) -> float | None:
    source = record.get("source") or {}
    if record["label"] == "live":
        return (source.get("path") or {}).get("sstd_db")
    return (source.get("playback") or {}).get("acoustic_sstd_db")


def _read_rows(manifest: Path, limit: int | None) -> list[dict]:
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
    return rows


def _collect(rows: list[dict], root: Path) -> dict:
    """Per-file features and levels, grouped the ways the report slices them."""
    out: dict = {
        "table": {"live": defaultdict(list), "device": defaultdict(list)},
        "by_env": defaultdict(lambda: {"live": [], "device": []}),
        "by_origin": defaultdict(lambda: {"live": [], "device": []}),
        "by_form": defaultdict(list),
        "sstd": defaultdict(list),
        "pairs": defaultdict(dict),
    }
    for record in rows:
        audio, sr = sf.read(str(root / record["audio"]), always_2d=False)
        feats = features(audio, sr)
        label = record["label"]
        for name, value in feats.items():
            out["table"][label][name].append(value)
        level = feats["level_dbfs"]
        out["by_env"][record["scene"]["environment"]][label].append(level)
        out["by_origin"][record.get("origin", "dry")][label].append(level)
        form = record.get("playback_form", "single")
        if label == "device":
            out["by_form"][form].append(level)
        spread = _acoustic_sstd(record)
        if spread is not None and np.isfinite(spread):
            out["sstd"]["live" if label == "live" else f"device:{form}"].append(float(spread))
        out["pairs"][f"{record['key']}#{record['variant']}"][label] = level
    return out


def run_audit(manifest: str | Path, limit: int | None = None) -> dict:
    manifest = Path(manifest)
    groups = _collect(_read_rows(manifest, limit), manifest.parent)
    table, by_env, by_origin = groups["table"], groups["by_env"], groups["by_origin"]
    by_form, sstd, pairs = groups["by_form"], groups["sstd"], groups["pairs"]

    result: dict = {
        "n_live": len(table["live"]["level_dbfs"]),
        "n_device": len(table["device"]["level_dbfs"]),
        "features": {},
        "level_auc_by_environment": {},
        "level_auc_by_playback_form": {},
        "level_auc_by_origin": {},
        "acoustic_sstd_db": {},
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
    live_levels = np.asarray(table["live"]["level_dbfs"])
    for env, values in sorted(by_env.items()):
        result["level_auc_by_environment"][env] = round(
            auc(np.asarray(values["device"]), np.asarray(values["live"])), 3
        )
    for form, values in sorted(by_form.items()):
        result["level_auc_by_playback_form"][form] = {
            "n": len(values),
            "auc": round(auc(np.asarray(values), live_levels), 3),
        }
    for origin, values in sorted(by_origin.items()):
        result["level_auc_by_origin"][origin] = round(
            auc(np.asarray(values["device"]), np.asarray(values["live"])), 3
        )
    for group, values in sorted(sstd.items()):
        result["acoustic_sstd_db"][group] = {
            "n": len(values),
            "median": round(float(np.median(values)), 2),
            "p10": round(float(np.percentile(values, 10)), 2),
            "p90": round(float(np.percentile(values, 90)), 2),
        }
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


def _auc_cells(table: dict) -> str:
    return ", ".join(f"{k} {v:.2f}" for k, v in table.items() if np.isfinite(v))


def format_report(result: dict) -> str:
    lines = [f"{result['n_live']} live clips, {result['n_device']} device clips", ""]
    lines.append(f"{'feature':<20}{'kind':<10}{'AUC':>7}{'live':>12}{'device':>12}")
    lines.append("-" * 61)
    for name, row in result["features"].items():
        flag = "  <- possible shortcut" if row["shortcut"] else ""
        live = "-" if row["live_median"] is None else f"{row['live_median']:.3f}"
        device = "-" if row["device_median"] is None else f"{row['device_median']:.3f}"
        lines.append(
            f"{name:<20}{row['kind']:<10}{row['auc_device_gt_live']:>7.3f}"
            f"{live:>12}{device:>12}{flag}"
        )
    lines.append("")
    lines.append("level AUC by environment: " + _auc_cells(result["level_auc_by_environment"]))
    if result["level_auc_by_playback_form"]:
        cells = [
            f"{form} {v['auc']:.2f} (n={v['n']})"
            for form, v in result["level_auc_by_playback_form"].items()
        ]
        lines.append("level AUC by playback form (vs all live): " + ", ".join(cells))
    if len(result["level_auc_by_origin"]) > 1:
        lines.append("level AUC by input origin: " + _auc_cells(result["level_auc_by_origin"]))
    if "paired_level_difference_db" in result:
        d = result["paired_level_difference_db"]
        lines.append(
            f"paired level difference device - live: mean {d['mean']:+.1f} dB, sd {d['std']:.1f}, "
            f"10-90% [{d['p10']:+.1f}, {d['p90']:+.1f}] over {d['n']} pairs"
        )
    if result["acoustic_sstd_db"]:
        cells = [
            f"{group} {v['median']:.1f} [{v['p10']:.1f}, {v['p90']:.1f}]"
            for group, v in result["acoustic_sstd_db"].items()
        ]
        lines.append("acoustic spectral spread, median [10-90%] dB: " + ", ".join(cells))
    lines.append("")
    lines.append(
        f"Nuisance features should stay within 0.5 +/- {SHORTCUT_MARGIN}; acoustic features are "
        "expected to separate the classes. One room spreads a transfer function by ~5.6 dB, "
        "two rooms in series by ~8 dB."
    )
    return "\n".join(lines)


__all__ = ["auc", "features", "format_report", "run_audit"]
