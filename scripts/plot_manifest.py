"""Plot what a rendered batch covers, split by class.

    python scripts/plot_manifest.py out/manifest.jsonl --out coverage.png

Works on the manifest written by ``render``. The panels are the things a rejection
model could latch onto if they differed between the classes - level at the
microphone, signal-to-noise ratio, distance - next to what should differ: which
devices and programme chains the device class was drawn from. The first row should
show two overlapping distributions; if the device class sits apart, retune the
device ``level_dba`` presets or the talker ``level_dba`` range.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOURS = {"live": "tab:green", "device": "tab:red"}


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [r for r in (json.loads(line) for line in handle if line.strip()) if "label" in r]


def distance(record: dict) -> float:
    source = record["source"]
    if record["label"] == "live":
        return source["path"]["distance_m"]
    return min(p["distance_m"] for p in source["paths"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, default=Path("coverage.png"))
    args = parser.parse_args()

    rows = load(args.manifest)
    by_label = {label: [r for r in rows if r["label"] == label] for label in COLOURS}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    panels = [
        ("speech level at the mic (dB(A))", lambda r: r["at_mic"]["speech_dba"]),
        ("SNR at the mic (dB)", lambda r: r["at_mic"]["snr_db"]),
        ("source-to-mic distance (m)", distance),
    ]
    for ax, (title, value) in zip(axes[0], panels, strict=True):
        for label, records in by_label.items():
            if records:
                ax.hist(
                    [value(r) for r in records],
                    bins=30,
                    alpha=0.5,
                    color=COLOURS[label],
                    label=f"{label} (n={len(records)})",
                )
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    devices = by_label["device"]
    counters = [
        ("environment", Counter(r["scene"]["environment"] for r in rows if r["label"] == "live")),
        ("device", Counter(r["source"]["device"] for r in devices)),
        (
            "programme / codec",
            Counter(
                f"{r['source']['content']['content']}/{r['source']['content'].get('codec', 'none')}"
                for r in devices
            ),
        ),
    ]
    for ax, (title, counts) in zip(axes[1], counters, strict=True):
        labels = [k for k, _ in counts.most_common(14)]
        ax.barh(labels[::-1], [counts[k] for k in labels[::-1]], color="tab:blue")
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.25, axis="x")

    fig.suptitle(f"Rendered batch coverage - {args.manifest}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
