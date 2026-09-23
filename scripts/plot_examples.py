"""Spectrograms of rendered examples.

    python scripts/plot_examples.py pairs demo/rendered --out demo/spectrograms.jpg
    python scripts/plot_examples.py scene demo/scenes --out demo/scene.jpg

``pairs`` shows, for a few utterances, the dry TTS next to its live-talker and
device-playback renders from the same scene - one row per device type found.
``scene`` shows a full-duplex scene with its live (green) and device (red) segments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal


def _load(root: Path) -> list[dict]:
    lines = (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    return [r for r in (json.loads(line) for line in lines if line.strip()) if "error" not in r]


def _save(fig, out: Path) -> None:
    """JPEG for spectrograms: PNG compresses their noise-like texture badly."""
    fig.tight_layout()
    jpeg = out.suffix.lower() in {".jpg", ".jpeg"}
    fig.savefig(out, dpi=85, pil_kwargs={"quality": 88} if jpeg else None)


def _spectrogram(ax, x: np.ndarray, sr: int, title: str) -> None:
    f, t, s = signal.stft(x, fs=sr, nperseg=512, noverlap=384, boundary=None)
    db = 20 * np.log10(np.abs(s) + 1e-9)
    ax.pcolormesh(t, f / 1000, db, vmin=db.max() - 75, vmax=db.max(), shading="auto", cmap="magma")
    ax.set_ylim(0, min(8.0, sr / 2000))
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("kHz", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_pairs(root: Path, out: Path, rows: int) -> None:
    pairs: dict[str, dict[str, dict]] = {}
    for record in _load(root):
        pairs.setdefault(record["pair"], {})[record["label"]] = record
    chosen, seen = [], set()
    for pair in pairs.values():
        if {"live", "device"} <= set(pair) and pair["device"]["source"]["device"] not in seen:
            seen.add(pair["device"]["source"]["device"])
            chosen.append(pair)
        if len(chosen) == rows:
            break
    fig, axes = plt.subplots(len(chosen), 3, figsize=(15, 2.9 * len(chosen)), squeeze=False)
    for ax_row, pair in zip(axes, chosen, strict=True):
        live, device = pair["live"], pair["device"]
        dry, sr_dry = sf.read(live["input"])
        scene = live["scene"]
        content = device["source"]["content"]
        _spectrogram(ax_row[0], dry, sr_dry, f"dry TTS: {live['key']}")
        _spectrogram(
            ax_row[1],
            *sf.read(root / live["audio"]),
            f"live talker - {scene['environment']}, {live['source']['path']['distance_m']} m, "
            f"{live['at_mic']['speech_dba']:.0f} dB(A), noise {scene['noise_dba']:.0f} dB(A)",
        )
        chain = f"{content['content']}/{content.get('codec', 'no codec')}"
        _spectrogram(
            ax_row[2],
            *sf.read(root / device["audio"]),
            f"{device['source']['device']} - {chain}, {device['at_mic']['speech_dba']:.0f} dB(A)",
        )
    _save(fig, out)


def plot_scene(root: Path, out: Path, index: int | None) -> None:
    records = _load(root)
    if index is None:
        record = next(
            (r for r in records if {s["role"] for s in r["segments"]} == {"live", "device"}),
            records[0],
        )
    else:
        record = records[index]
    x, sr = sf.read(root / record["audio"])
    fig, ax = plt.subplots(figsize=(15, 3.6))
    _spectrogram(ax, x, sr, "")
    sources = ", ".join(
        f"{k}: {v.get('device', 'live talker')}" for k, v in record["sources"].items()
    )
    ax.set_title(
        f"{record['scene_id']} - {record['scene']['environment']} - {sources} "
        f"(green = live user, red = device)",
        fontsize=9,
    )
    lanes = {"device": (0.0, 0.035, "red"), "live": (0.045, 0.08, "lime")}
    for s in record["segments"]:
        low, high, colour = lanes[s["role"]]
        ax.axvspan(s["start"], s["end"], ymin=low, ymax=high, color=colour, alpha=0.9)
    ax.set_xlim(0, record["duration_s"])
    ax.set_xlabel("s", fontsize=8)
    _save(fig, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["pairs", "scene"])
    parser.add_argument("root", type=Path, help="output directory of render / scenes")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4, help="pairs: number of rows")
    parser.add_argument("--index", type=int, help="scene: manifest line to plot")
    args = parser.parse_args()
    if args.what == "pairs":
        plot_pairs(args.root, args.out, args.rows)
    else:
        plot_scene(args.root, args.out, args.index)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
