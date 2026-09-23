"""Compare what a spoofing detector says about each stage of the simulated chain.

    python scripts/detector_report.py \
        --scores clean=scores_clean.txt \
        --scores room=scores_room.txt \
        --scores chain=scores_chain.txt \
        --scores real=scores_replaydf.txt \
        --baseline clean --reference real

A single detector score is not evidence that the simulation is realistic: every
detector here was trained to separate clean close-talk from re-recorded audio, so
reverb-only audio also scores as "replayed". What carries information is the
*comparison* - how far each condition moves the detector relative to clean audio,
and how much of the way to a real re-recording it gets.

Score files have one trial per line, ``path<TAB>label<TAB>score``, where label is
bonafide or spoof. The default convention is the one used by every checkpoint in
the README: a higher score means more bona fide.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BONAFIDE = {"bonafide", "bona_fide", "bona-fide", "genuine", "real", "target", "1"}
SPOOF = {"spoof", "spoofed", "fake", "nontarget", "0"}


def read_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``path<TAB>label<TAB>score`` into bona fide and spoof score arrays."""
    bonafide: list[float] = []
    spoof: list[float] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t") if "\t" in line else line.rsplit(maxsplit=2)
        if len(fields) != 3:
            raise ValueError(f"{path}:{lineno}: expected 'path<TAB>label<TAB>score', got {line!r}")
        label, score = fields[1].strip().lower(), float(fields[2])
        if label in BONAFIDE:
            bonafide.append(score)
        elif label in SPOOF:
            spoof.append(score)
        else:
            raise ValueError(f"{path}:{lineno}: unknown label {fields[1]!r}")
    return np.asarray(bonafide), np.asarray(spoof)


def equal_error_rate(bonafide: np.ndarray, spoof: np.ndarray) -> float:
    """EER in percent, assuming a higher score means more bona fide."""
    if bonafide.size == 0 or spoof.size == 0:
        raise ValueError("need both bona fide and spoof trials to compute an EER")
    scores = np.concatenate([bonafide, spoof])
    labels = np.concatenate([np.ones(bonafide.size), np.zeros(spoof.size)])
    labels = labels[np.argsort(scores, kind="mergesort")]
    # Sweep the decision threshold up through the sorted scores: misses are the
    # bona fide trials already below it, false alarms the spoof trials still above.
    miss = np.concatenate([[0.0], np.cumsum(labels) / bonafide.size])
    false_alarm = np.concatenate([[1.0], 1.0 - np.cumsum(1.0 - labels) / spoof.size])
    index = int(np.argmin(np.abs(miss - false_alarm)))
    return float(50.0 * (miss[index] + false_alarm[index]))


def class_accuracy(scores: np.ndarray, threshold: float, is_bonafide: bool) -> float:
    """Percentage of trials the detector puts on the correct side of ``threshold``."""
    if scores.size == 0:
        return float("nan")
    correct = scores > threshold if is_bonafide else scores <= threshold
    return float(100.0 * np.mean(correct))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a named score file; repeat for each condition",
    )
    parser.add_argument("--baseline", help="condition to measure the others against")
    parser.add_argument("--reference", help="condition holding real re-recordings")
    parser.add_argument(
        "--threshold",
        type=float,
        help="decision threshold, to also report per-class accuracy",
    )
    parser.add_argument(
        "--higher-is-spoof",
        action="store_true",
        help="flip the score convention",
    )
    args = parser.parse_args()

    conditions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for item in args.scores:
        if "=" not in item:
            parser.error(f"--scores expects NAME=PATH, got {item!r}")
        name, _, path = item.partition("=")
        bonafide, spoof = read_scores(Path(path))
        if args.higher_is_spoof:
            bonafide, spoof = -bonafide, -spoof
        conditions[name] = (bonafide, spoof)

    names = list(conditions)
    baseline = args.baseline or names[0]
    for role, name in (("--baseline", baseline), ("--reference", args.reference)):
        if name is not None and name not in conditions:
            parser.error(f"{role}={name!r} is not one of {names}")

    eers = {name: equal_error_rate(*pair) for name, pair in conditions.items()}
    base_eer = eers[baseline]
    real_gap = eers[args.reference] - base_eer if args.reference else None

    header = f"{'condition':<14}{'bonafide':>9}{'spoof':>8}{'EER %':>9}{'dEER':>8}"
    if real_gap:
        header += f"{'gap closed':>12}"
    if args.threshold is not None:
        header += f"{'acc bona %':>12}{'acc spoof %':>13}"
    print(header)
    print("-" * len(header))

    for name, (bonafide, spoof) in conditions.items():
        row = (
            f"{name:<14}{bonafide.size:>9}{spoof.size:>8}"
            f"{eers[name]:>9.2f}{eers[name] - base_eer:>+8.2f}"
        )
        if real_gap:
            share = 100.0 * (eers[name] - base_eer) / real_gap
            row += "         ---" if name == baseline else f"{share:>11.0f}%"
        if args.threshold is not None:
            row += f"{class_accuracy(bonafide, args.threshold, True):>12.1f}"
            row += f"{class_accuracy(spoof, args.threshold, False):>13.1f}"
        print(row)

    print()
    print(f"baseline: {baseline}")
    if args.reference:
        print(
            f"'gap closed' is how much of the {baseline} -> {args.reference} EER gap "
            f"({real_gap:+.2f} points) each condition reproduces."
        )
        print("A full chain that only reproduces what the room-only row does is just reverb.")
    print(
        "Reference points from ReplayDF (Interspeech 2025), real re-recording of the same "
        "audio: W2V2-AASIST public checkpoint 10.6% -> 24.8% EER; retrained on ODSS "
        "4.7% -> 18.2%. Spoof accuracy fell 28-42 points while bona fide stayed ~98%."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
