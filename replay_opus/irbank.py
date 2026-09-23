"""Impulse response sourcing: measured files, parametric synthesis, room simulation.

Three ways to obtain the room stage, in descending order of fidelity:

1. ``measured``  - convolve a real IR (ReplayDF, MIT IR Survey, SLR28, your own sweeps).
2. ``synthetic`` - reshape a measured IR to a requested (T60, DRR) pair, which is how
   Luong et al. (SLT 2024) grew 270 measured IRs into a set 500x larger.
3. ``simulated`` - image-source method via pyroomacoustics when no measured IR exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .dsp import EPS, resample_to, to_mono
from .metrics import (
    describe_ir,
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
    that was never recorded. Targets above the IR's own T60 are clamped, so build long
    tails with :func:`synth_ir` instead.
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

    direct = np.zeros(length)
    direct[0] = 1.0
    # Suppress the very beginning of the late field so the direct path stays isolated.
    late[: max(1, int(sr * 0.0025))] = 0.0

    h = retarget_t60(direct + late, sr, t60)
    h = set_drr(h, sr, drr)
    norm = np.sqrt(np.sum(h**2))
    return h / (norm + EPS)


def simulate_room_ir(
    sr: int,
    room_dim: tuple[float, float, float],
    src: tuple[float, float, float],
    mic: tuple[float, float, float],
    t60: float,
    max_order: int | None = None,
) -> np.ndarray:
    """Image-source RIR via pyroomacoustics. Optional dependency.

    ``max_order`` defaults to the order pyroomacoustics says is needed for the
    requested T60; capping it below that truncates the tail and the realised T60
    comes out short.
    """
    try:
        import pyroomacoustics as pra  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "room_mode='simulated' needs pyroomacoustics (pip install pyroomacoustics)"
        ) from exc

    absorption, computed_order = pra.inverse_sabine(t60, room_dim)
    order = computed_order if max_order is None else min(max_order, computed_order)
    room = pra.ShoeBox(
        room_dim,
        fs=sr,
        materials=pra.Material(absorption),
        max_order=order,
    )
    room.add_source(list(src))
    room.add_microphone(list(mic))
    room.compute_rir()
    return prepare_ir(np.asarray(room.rir[0][0], dtype=np.float64), sr)


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
        return self.get(self.paths[index]), self._label(index)

    def _label(self, index: int) -> str:
        if self.labels:
            return self.labels[index]
        path = self.paths[index]
        if self.root is not None:
            return path.relative_to(self.root).as_posix()
        return path.name

    def stats(self) -> list[dict]:
        rows = []
        for index, path in enumerate(self.paths):
            label = self._label(index)
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
    "describe_ir",
    "estimate_drr",
    "estimate_t60",
    "filter_ir_stats",
    "load_ir",
    "prepare_ir",
    "reshape_ir",
    "retarget_t60",
    "set_drr",
    "simulate_room_ir",
    "synth_ir",
    "trim_noise_floor",
]
