"""Measured impulse responses and parametric IR synthesis.

Used for three things: measured room IRs that stand in for the geometric room model
(grouped by room so that the talker and the devices of one scene come from the same
real room), measured microphone responses, and the small parametric rooms that
content was originally recorded in.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .dsp import EPS, resample_to, to_mono
from .metrics import (
    describe_ir,
    direct_index,
    energy_decay_curve,
    estimate_drr,
    estimate_t60,
    split_direct_late,
)
from .pathlist import common_root, read_path_list, relative_labels

IR_SUFFIXES = (".wav", ".flac", ".ogg", ".aiff", ".aif")


def amplitude_decay_rate(t60: float) -> float:
    """Envelope constant ``a`` such that ``exp(-a*T60)`` equals -60 dB."""
    return 3.0 * np.log(10.0) / max(t60, 1e-3)


def load_ir(path: str | Path, sr: int, max_ms: float = 1500.0) -> np.ndarray:
    h, sr_file = sf.read(str(path), always_2d=False)
    h = to_mono(h)
    h = resample_to(h, sr_file, sr)
    return prepare_ir(h, sr, max_ms=max_ms)


def prepare_ir(h: np.ndarray, sr: int, max_ms: float = 1500.0, pre_ms: float = 1.0) -> np.ndarray:
    """Trim the silence before the direct path, cap the tail, normalize the energy.

    Without the leading trim, every IR would inject its own random latency; without
    the energy normalization, the IR would double as a random gain.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size == 0:
        raise ValueError("empty impulse response")
    d0 = int(np.argmax(np.abs(h)))
    start = max(0, d0 - int(sr * pre_ms / 1000.0))
    h = h[start : start + int(sr * max_ms / 1000.0)]
    norm = np.sqrt(np.sum(h**2))
    return h / (norm + EPS)


def trim_noise_floor(h: np.ndarray, sr: int, floor_db: float = -70.0) -> np.ndarray:
    """Cut the IR where its decay curve reaches the measurement noise floor.

    Required before shortening a measured IR: the envelope rescaling would otherwise
    lift the recorded noise floor and flatten the decay, inflating the T60 estimate.
    """
    d0 = int(np.argmax(np.abs(h)))
    edc = energy_decay_curve(h[d0:])
    below = np.flatnonzero(edc <= floor_db)
    if below.size == 0:
        return h
    return h[: d0 + int(below[0]) + 1]


def retarget_t60(
    h: np.ndarray, sr: int, target_t60: float, iters: int = 6, tol: float = 0.01
) -> np.ndarray:
    """Scale the late field's decay envelope until the measured T60 hits the target.

    Iterative because band-dependent decay and truncation make the single-shot
    envelope correction only approximately right.
    """
    _, _, _, late_start = split_direct_late(h, sr)
    t = np.maximum((np.arange(len(h)) - late_start) / sr, 0.0)
    out = np.asarray(h, dtype=np.float64).copy()
    for _ in range(iters):
        current = estimate_t60(out, sr)
        if not np.isfinite(current) or current <= 0:
            break
        if abs(current - target_t60) / target_t60 < tol:
            break
        delta = amplitude_decay_rate(target_t60) - amplitude_decay_rate(current)
        out = out * np.exp(-delta * t)
    return out


def set_drr(h: np.ndarray, sr: int, target_drr: float) -> np.ndarray:
    """Rebalance direct and late energy to a target DRR, keeping total energy fixed."""
    direct, late, _, _ = split_direct_late(h, sr)
    e_direct = float(np.sum(direct**2))
    e_late = float(np.sum(late**2))
    if e_direct < EPS or e_late < EPS:
        return h
    total = e_direct + e_late
    ratio = 10.0 ** (target_drr / 10.0)
    want_direct = total * ratio / (1.0 + ratio)
    return direct * np.sqrt(want_direct / e_direct) + late * np.sqrt((total - want_direct) / e_late)


def reshape_ir(
    h: np.ndarray,
    sr: int,
    target_t60: float | None = None,
    target_drr: float | None = None,
) -> np.ndarray:
    """Retarget a measured IR's T60 and DRR (Bryan, ICASSP 2020).

    T60 can only be *shortened*: lengthening would have to invent a reverberant tail
    that was never recorded. Targets above the IR's own T60 are clamped.
    """
    out = trim_noise_floor(np.asarray(h, dtype=np.float64), sr)

    if target_t60 is not None:
        current = estimate_t60(out, sr)
        if np.isfinite(current) and current > 0:
            out = retarget_t60(out, sr, min(target_t60, current))

    if target_drr is not None:
        out = set_drr(out, sr, target_drr)

    norm = np.sqrt(np.sum(out**2))
    return out / (norm + EPS)


def synth_ir(
    sr: int,
    t60: float,
    drr: float,
    rng: np.random.Generator,
    n_bands: int = 6,
    hf_decay_factor: float = 0.6,
) -> np.ndarray:
    """Build an IR from scratch as a direct path plus band-wise decaying noise.

    Higher octave bands are given shorter decay times (``hf_decay_factor``) to mimic
    air absorption and soft-surface damping, which a single-band exponential misses.
    Because that band spread pulls the broadband T60 below any single band's value,
    the bands are built long and then trimmed back to the requested T60.
    """
    nominal = t60 * 1.6
    length = max(int(sr * min(nominal * 1.6 + 0.05, 5.0)), int(sr * 0.05))
    t = np.arange(length) / sr
    edges = np.geomspace(125.0, min(0.95 * sr / 2, 8000.0), n_bands + 1)

    late = np.zeros(length)
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        band_t60 = nominal * (hf_decay_factor ** (i / max(n_bands - 1, 1)))
        noise = rng.standard_normal(length)
        sos = signal.butter(
            4, (lo / (sr / 2), min(hi / (sr / 2), 0.999)), btype="band", output="sos"
        )
        late += signal.sosfilt(sos, noise) * np.exp(-amplitude_decay_rate(band_t60) * t)

    # Suppress the very beginning of the late field so the direct path stays isolated.
    late[: max(1, int(sr * 0.0025))] = 0.0
    # Shape and scale the late field on its own: before scaling, its noise peaks can
    # exceed the unit direct path, and anything locating the direct by its maximum
    # would then pick a late-field peak and throw the real direct sound away.
    late = retarget_t60(late, sr, t60)
    late *= np.sqrt(10.0 ** (-drr / 10.0) / (float(np.sum(late**2)) + EPS))
    h = late
    h[0] += 1.0
    norm = np.sqrt(np.sum(h**2))
    return h / (norm + EPS)


def scale_for_distance(h: np.ndarray, sr: int, distance_m: float) -> np.ndarray:
    """Give a measured IR the direct-path amplitude of a source ``distance_m`` away.

    A measured IR's absolute gain means nothing across datasets. Pinning the direct
    part to ``1/d`` keeps the IR's own direct-to-reverberant balance while putting the
    source at a physically plausible level for where the scene placed it.
    """
    direct, _, _, _ = split_direct_late(h, sr)
    e_direct = float(np.sum(direct**2))
    return h / (np.sqrt(e_direct) + EPS) / max(distance_m, 0.05)


def graft_tail(
    modelled: np.ndarray, measured: np.ndarray, sr: int, split_s: float, fade_s: float = 0.005
) -> np.ndarray:
    """Keep ``modelled`` up to ``split_s`` after its direct path, then continue with the
    measured IR's late field scaled to the modelled late energy.

    A measured IR was recorded from a measurement loudspeaker at one position, so its
    direct sound and early reflections belong to that loudspeaker and that spot. The
    modelled head keeps what depends on the source actually placed in the scene - its
    distance, its directivity (a mouth is not a loudspeaker), its early reflections -
    and the late energy that goes with them, so the direct-to-reverberant ratio still
    follows the scene. The measured tail brings the real room's decay and colour.
    """
    modelled = np.asarray(modelled, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    split = round(split_s * sr)
    fade = max(1, round(fade_s * sr))
    seam = direct_index(modelled) + split
    source = direct_index(measured) + split
    if seam >= len(modelled) or source + fade >= len(measured) or seam < fade:
        return modelled
    e_model = float(np.sum(modelled[seam:] ** 2))
    e_tail = float(np.sum(measured[source:] ** 2))
    if e_model < EPS or e_tail < EPS:
        return modelled
    tail = measured[source - fade :] * np.sqrt(e_model / e_tail)
    ramp = np.sin(0.5 * np.pi * (np.arange(fade) + 0.5) / fade) ** 2
    head = modelled[:seam].copy()
    head[-fade:] *= 1.0 - ramp
    tail[:fade] *= ramp
    out = np.zeros(max(len(head), seam - fade + len(tail)))
    out[: len(head)] += head
    out[seam - fade : seam - fade + len(tail)] += tail
    return out


@dataclass
class IRBank:
    """A pool of impulse responses named by a path list, loaded lazily and cached."""

    root: Path | None
    sr: int
    max_ms: float = 1500.0
    paths: list[Path] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    # Bounded so pointing at a large simulated-RIR corpus (SLR28 has tens of
    # thousands) cannot grow the worker's memory without limit.
    cache_size: int = 512
    _cache: dict[Path, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def from_list(cls, list_path: str | Path | None, sr: int, max_ms: float = 1500.0) -> IRBank:
        if list_path is None:
            return cls(root=None, sr=sr, max_ms=max_ms)
        paths = read_path_list(list_path, kind="impulse response", suffixes=IR_SUFFIXES)
        return cls(
            root=common_root(paths),
            sr=sr,
            max_ms=max_ms,
            paths=paths,
            labels=relative_labels(paths),
        )

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = load_ir(path, self.sr, self.max_ms)
        return self._cache[path]

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        if not self.paths:
            raise RuntimeError("IR bank is empty")
        index = int(rng.integers(len(self.paths)))
        return self.get(self.paths[index]), self.labels[index]

    def rooms(self) -> dict[str, list[int]]:
        """Indices grouped by parent directory: one directory per measured room."""
        groups: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            groups[str(Path(label).parent.as_posix())].append(index)
        return dict(groups)

    def stats(self) -> list[dict]:
        rows = []
        for index, path in enumerate(self.paths):
            label = self.labels[index]
            try:
                h = self.get(path)
            except Exception as exc:  # noqa: BLE001 - report and continue
                rows.append({"ir": label, "error": str(exc)})
                continue
            row = {"ir": label}
            row.update(describe_ir(h, self.sr))
            rows.append(row)
        return rows


def filter_ir_stats(rows: list[dict], t60_max: float = 2.0, drr_range=(-10.0, 30.0)) -> list[dict]:
    """Drop degenerate IRs, matching the exclusion rule used by Luong et al. (2024)."""
    keep = []
    for row in rows:
        t60, drr = row.get("t60_s"), row.get("drr_db")
        if t60 is None or drr is None or not np.isfinite(t60) or not np.isfinite(drr):
            continue
        if t60 > t60_max or not (drr_range[0] <= drr <= drr_range[1]):
            continue
        keep.append(row)
    return keep


__all__ = [
    "IRBank",
    "amplitude_decay_rate",
    "estimate_drr",
    "filter_ir_stats",
    "graft_tail",
    "load_ir",
    "prepare_ir",
    "reshape_ir",
    "retarget_t60",
    "scale_for_distance",
    "set_drr",
    "synth_ir",
    "trim_noise_floor",
]
