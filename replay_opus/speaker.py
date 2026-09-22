"""Playback stage: loudspeaker linear response plus non-linear distortion.

The model is a generalised polynomial Hammerstein system, ``y = sum_n g_n * x^n``,
which is what ASVspoof 2019 PA used for its replay devices and what the synchronised
swept-sine method in :mod:`replay_opus.measure` identifies from a single sweep.

Kernels come from either source:

* measured - an ``.npz`` written by ``replay-opus measure extract``.
* synthetic - :func:`synth_speaker_kernels`, parameterised by a quality label that
  mirrors the perfect / high / low device grades of ASVspoof 2019 PA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from .dsp import EPS, butter, peaking_eq, set_peak_db, soft_clip

QUALITY_PRESETS: dict[str, dict] = {
    # Flat and clean: a studio monitor. Equivalent to the "perfect" Dirac grade.
    "perfect": {
        "hp_hz": (20.0, 40.0),
        "lp_hz": (16000.0, 20000.0),
        "harmonic_rolloff_db": (-46.0, -38.0),
        "ripple_db": 1.0,
        "clip_kind": "none",
    },
    # A decent portable speaker or a laptop with real drivers.
    "high": {
        "hp_hz": (90.0, 180.0),
        "lp_hz": (11000.0, 16000.0),
        "harmonic_rolloff_db": (-34.0, -26.0),
        "ripple_db": 3.5,
        "clip_kind": "tanh",
    },
    # A phone earpiece or a cheap bluetooth puck: band-limited and audibly distorted.
    "low": {
        "hp_hz": (220.0, 420.0),
        "lp_hz": (5500.0, 9000.0),
        "harmonic_rolloff_db": (-22.0, -14.0),
        "ripple_db": 7.0,
        "clip_kind": "tanh",
    },
}


@dataclass
class SpeakerModel:
    """Hammerstein kernels ``g_n`` and the polynomial orders ``n`` they apply to."""

    kernels: list[np.ndarray]
    orders: list[int]
    sr: int
    label: str = "synthetic"
    clip_kind: str = "tanh"

    @classmethod
    def load(cls, path: str | Path, sr: int) -> SpeakerModel:
        data = np.load(str(path))
        orders = [int(o) for o in data["orders"]]
        kernels = [np.asarray(data[f"g{o}"], dtype=np.float64) for o in orders]
        stored_sr = int(data["sr"]) if "sr" in data else sr
        if stored_sr != sr:
            raise ValueError(
                f"speaker model was measured at {stored_sr} Hz but the chain runs at {sr} Hz"
            )
        return cls(
            kernels=kernels,
            orders=orders,
            sr=sr,
            label=str(data["label"]) if "label" in data else Path(path).stem,
            clip_kind=str(data["clip_kind"]) if "clip_kind" in data else "none",
        )

    def save(self, path: str | Path) -> None:
        payload = {f"g{o}": k for o, k in zip(self.orders, self.kernels, strict=True)}
        np.savez(
            str(path),
            orders=np.array(self.orders),
            sr=self.sr,
            label=self.label,
            clip_kind=self.clip_kind,
            **payload,
        )


def _bandpass_kernel(
    sr: int, hp_hz: float, lp_hz: float, ripple_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Impulse response of a band-limited, slightly rippled transducer.

    The kernel is sized from the high-pass corner: a 4th-order high-pass at 20 Hz
    rings for tens of milliseconds, and truncating that would misreport the very
    low-frequency rolloff the kernel exists to model.
    """
    taps = int(np.clip(8 * sr / max(hp_hz, 20.0), 512, 16_384))
    x = np.zeros(taps)
    x[taps // 8] = 1.0
    y = butter(x, sr, hp_hz, kind="high", order=4)
    y = butter(y, sr, min(lp_hz, 0.98 * sr / 2), kind="low", order=6)
    lo_edge, hi_edge = np.log(max(hp_hz, 60.0)), np.log(min(lp_hz, sr / 2 * 0.9))
    for _ in range(3):
        f0 = float(np.exp(rng.uniform(lo_edge, hi_edge)))
        gain = float(rng.uniform(-ripple_db, ripple_db))
        y = peaking_eq(y, sr, f0, gain, float(rng.uniform(0.7, 3.0)))
    return y


def synth_speaker_kernels(
    sr: int,
    rng: np.random.Generator,
    quality: str = "high",
    n_orders: int = 5,
) -> SpeakerModel:
    """Synthesise Hammerstein kernels for a device of the given quality grade.

    Five orders are enough: ASVspoof 2019 PA found the first five higher-harmonic
    responses capture the bulk of a replay device's non-linearity.
    """
    if quality not in QUALITY_PRESETS:
        raise ValueError(
            f"unknown speaker quality {quality!r}; choose from {list(QUALITY_PRESETS)}"
        )
    preset = QUALITY_PRESETS[quality]
    hp = float(rng.uniform(*preset["hp_hz"]))
    lp = float(rng.uniform(*preset["lp_hz"]))
    rolloff_db = float(rng.uniform(*preset["harmonic_rolloff_db"]))

    g1 = _bandpass_kernel(sr, hp, lp, preset["ripple_db"], rng)
    g1 = g1 / (np.max(np.abs(g1)) + EPS)

    orders = list(range(1, n_orders + 1))
    kernels = [g1]
    for n in orders[1:]:
        # Each higher order is weaker and duller than the one below it.
        gain = 10.0 ** (rolloff_db * (n - 1) / 20.0)
        kn = butter(g1, sr, min(lp / n, 0.98 * sr / 2), kind="low", order=4) * gain
        kernels.append(kn * float(rng.uniform(0.6, 1.4)))

    return SpeakerModel(
        kernels=kernels,
        orders=orders,
        sr=sr,
        label=f"synthetic:{quality}",
        clip_kind=preset["clip_kind"],
    )


def apply_hammerstein(x: np.ndarray, model: SpeakerModel) -> np.ndarray:
    """Evaluate ``sum_n g_n * x^n``.

    Even-order terms rectify the signal and so introduce DC; the caller is expected
    to high-pass the result (:func:`apply_speaker` does).
    """
    out = np.zeros(len(x) + max(len(k) for k in model.kernels) - 1)
    for order, kernel in zip(model.orders, model.kernels, strict=True):
        branch = signal.fftconvolve(np.power(x, order), kernel, mode="full")
        out[: len(branch)] += branch
    return out


def apply_speaker(
    x: np.ndarray,
    sr: int,
    model: SpeakerModel,
    rng: np.random.Generator,
    drive_db: float = -3.0,
    dc_block_hz: float = 30.0,
) -> tuple[np.ndarray, dict]:
    """Run the playback stage.

    Peak-normalising *before* the non-linearity is deliberate: distortion depends on
    absolute level, so without it the amount of distortion would be an accident of
    each input file's loudness. RIRplay v0.5 changed to exactly this order after
    finding clipping artefacts in its first release.
    """
    driven = set_peak_db(x, drive_db)
    y = apply_hammerstein(driven, model)
    y = soft_clip(y, model.clip_kind, drive=1.0)
    y = butter(y, sr, dc_block_hz, kind="high", order=2)
    return y, {
        "speaker": model.label,
        "drive_db": round(float(drive_db), 2),
        "orders": list(model.orders),
        "clip": model.clip_kind,
    }
