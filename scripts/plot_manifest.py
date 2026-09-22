"""Plot the acoustic coverage of a generated batch, optionally against a reference.

    python scripts/plot_manifest.py out/manifest.jsonl --out coverage.png
    python scripts/plot_manifest.py out/manifest.jsonl --reference real_irs.jsonl

The reference file is the JSONL written by ``replay_opus ir-stats --output``, so you
can check that your simulated (T60, DRR) distribution actually covers the real
recordings you are trying to imitate rather than sitting in a corner of the space.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def acoustic_points(rows: list[dict]) -> tuple[list[float], list[float]]:
    t60, drr = [], []
    for row in rows:
        measured = row.get("room_ir_measured", row)
        if "t60_s" in measured and "drr_db" in measured:
            t60.append(measured["t60_s"])
            drr.append(measured["drr_db"])
    return t60, drr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--reference", type=Path, help="ir-stats JSONL to overlay")
    parser.add_argument("--out", type=Path, default=Path("coverage.png"))
    args = parser.parse_args()

    rows = load(args.manifest)
    t60, drr = acoustic_points(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    if args.reference:
        ref_t60, ref_drr = acoustic_points(load(args.reference))
        ax.scatter(ref_t60, ref_drr, s=18, alpha=0.55, label=f"reference (n={len(ref_t60)})")
    ax.scatter(t60, drr, s=14, alpha=0.7, label=f"generated (n={len(t60)})")
    ax.set_xlabel("T60 (s)")
    ax.set_ylabel("DRR (dB)")
    ax.set_title("Acoustic coverage")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    snr = [r["snr_db"] for r in rows if "snr_db" in r]
    if snr:
        ax.hist(snr, bins=24)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("files")
    ax.set_title("Noise level")
    ax.grid(alpha=0.25)

    ax = axes[2]
    counts = Counter(str(r.get("codec")) for r in rows)
    labels = list(counts)
    ax.bar(labels, [counts[k] for k in labels])
    ax.set_ylabel("files")
    ax.set_title("Codec condition")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"Replay condition coverage - {args.manifest}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
