"""Ambient noise shared by every source in a scene, and a bank for audio beds.

Home conditions: quiet room tone, HVAC / fan, appliance hum, street traffic heard
through a window. Car conditions: idle, city and highway driving, mixed from road,
engine-order, wind and blower components. Real recordings named by a path list
take precedence; the synthetic recipes let the pipeline run with no downloads.

Levels are A-weighted SPL at the microphone, so the talker, the playback devices and
the noise share one physical scale, and one scene's noise is the same realisation
whichever source is rendered into it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .dsp import EPS, P_REF, a_weight, resample_to, rms, to_mono
from .pathlist import read_path_list, relative_labels

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg")


def shaped_noise(n: int, sr: int, rng: np.random.Generator, psd_db) -> np.ndarray:
    """Gaussian noise whose power spectrum follows ``psd_db(freqs)``."""
    n_fft = max(n, 16)
    spectrum = np.fft.rfft(rng.standard_normal(n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    shaped = np.fft.irfft(spectrum * 10.0 ** (psd_db(np.maximum(freqs, 1.0)) / 20.0), n_fft)[:n]
    return shaped / (np.std(shaped) + EPS)


def _slope(freqs, corner, db_per_octave):
    return db_per_octave * np.log2(np.maximum(freqs / corner, 1.0))


def _bump(freqs, centre, gain_db, octaves):
    return gain_db * np.exp(-0.5 * (np.log2(freqs / centre) / octaves) ** 2)


def _harmonics(
    n: int,
    sr: int,
    rng: np.random.Generator,
    f0: float,
    count: int,
    rolloff_db: float,
    wobble: float = 0.01,
) -> np.ndarray:
    """A harmonic series whose fundamental drifts slowly, like a motor under load."""
    t = np.arange(n) / sr
    drift = 1.0 + wobble * np.sin(
        2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.uniform(0, 2 * np.pi)
    )
    phase = 2 * np.pi * np.cumsum(f0 * drift) / sr
    out = np.zeros(n)
    for k in range(1, count + 1):
        if k * f0 * (1 + wobble) >= 0.45 * sr:
            break
        out += 10.0 ** (-rolloff_db * (k - 1) / 20.0) * np.sin(
            k * phase + rng.uniform(0, 2 * np.pi)
        )
    return out / (np.std(out) + EPS)


def _unit_a(x: np.ndarray, sr: int) -> np.ndarray:
    """Scale to unit A-weighted RMS so components mix by their audible weight."""
    return x / (rms(a_weight(x, sr)) + EPS)


def _home(condition: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    room_tone = shaped_noise(n, sr, rng, lambda f: -_slope(f, 80.0, 3.0) - _slope(f, 4000.0, 6.0))
    parts = [_unit_a(room_tone, sr)]
    if condition == "hvac":
        fan = shaped_noise(
            n, sr, rng, lambda f: _bump(f, rng.uniform(250, 700), 8.0, 1.2) - _slope(f, 1500.0, 4.0)
        )
        blade = _harmonics(n, sr, rng, rng.uniform(40.0, 180.0), 4, 8.0, 0.004)
        parts += [
            _unit_a(fan, sr) * 10 ** (rng.uniform(4, 10) / 20),
            _unit_a(blade, sr) * 10 ** (rng.uniform(-4, 4) / 20),
        ]
    elif condition == "appliance":
        mains = 50.0 if rng.random() < 0.6 else 60.0
        hum = _harmonics(n, sr, rng, 2 * mains, 6, 5.0, 0.0005)
        motor = shaped_noise(
            n,
            sr,
            rng,
            lambda f: _bump(f, rng.uniform(300, 1500), 6.0, 1.5) - _slope(f, 2000.0, 5.0),
        )
        parts += [
            _unit_a(hum, sr) * 10 ** (rng.uniform(0, 8) / 20),
            _unit_a(motor, sr) * 10 ** (rng.uniform(0, 8) / 20),
        ]
    elif condition == "street":
        traffic = shaped_noise(n, sr, rng, lambda f: -_slope(f, 90.0, 6.0) - _slope(f, 600.0, 6.0))
        t = np.arange(n) / sr
        swell = 1.0 + 0.6 * np.clip(
            np.sin(2 * np.pi * rng.uniform(0.05, 0.15) * t + rng.uniform(0, 6.3)), 0, None
        )
        parts += [_unit_a(traffic * swell, sr) * 10 ** (rng.uniform(4, 10) / 20)]
    return sum(parts)


def _car(condition: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    rpm = {"idle": (650, 900), "city": (1200, 2500), "highway": (1800, 3000)}[condition]
    cylinders = float(rng.choice([3, 4, 4, 4, 6]))
    firing = rng.uniform(*rpm) / 60.0 * cylinders / 2.0
    engine = _harmonics(n, sr, rng, firing, 10, rng.uniform(3.0, 6.0), 0.015)
    road = shaped_noise(
        n,
        sr,
        rng,
        lambda f: (
            -_slope(f, rng.uniform(60, 120), 7.0)
            - _slope(f, 1000.0, 3.0)
            + _bump(f, rng.uniform(190, 250), 4.0, 0.25)
        ),
    )
    wind = shaped_noise(n, sr, rng, lambda f: -_slope(f, 250.0, 3.0) - _slope(f, 5000.0, 6.0))
    fan = shaped_noise(
        n, sr, rng, lambda f: _bump(f, rng.uniform(400, 1200), 6.0, 1.3) - _slope(f, 2500.0, 4.0)
    )
    mix_db = {
        "idle": {"engine": (0, 0), "fan": (-6, 3), "road": (-20, -12), "wind": (-30, -25)},
        "city": {"engine": (-8, -2), "fan": (-12, 0), "road": (0, 0), "wind": (-16, -8)},
        "highway": {"engine": (-12, -5), "fan": (-14, -3), "road": (0, 0), "wind": (-6, 0)},
    }[condition]
    parts = {"engine": engine, "road": road, "wind": wind, "fan": fan}
    return sum(_unit_a(parts[k], sr) * 10 ** (rng.uniform(*mix_db[k]) / 20) for k in parts)


HOME_CONDITIONS = ("quiet", "hvac", "appliance", "street")
CAR_CONDITIONS = ("idle", "city", "highway")


def synth_noise(
    family: str, condition: str, n: int, sr: int, rng: np.random.Generator
) -> np.ndarray:
    if family == "car":
        if condition not in CAR_CONDITIONS:
            raise ValueError(
                f"unknown car noise condition {condition!r}; choose from {CAR_CONDITIONS}"
            )
        return _car(condition, n, sr, rng)
    if condition not in HOME_CONDITIONS:
        raise ValueError(
            f"unknown home noise condition {condition!r}; choose from {HOME_CONDITIONS}"
        )
    return _home(condition, n, sr, rng)


def scale_to_dba(x: np.ndarray, sr: int, level_dba: float) -> np.ndarray:
    """Scale a signal so its A-weighted level is ``level_dba`` SPL (output in pascals)."""
    current = rms(a_weight(x, sr))
    return x * (P_REF * 10.0 ** (level_dba / 20.0) / (current + EPS))


@dataclass
class AudioBank:
    """Recordings named by a path list, loaded lazily; random segments on demand."""

    sr: int
    paths: list[Path] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    cache_size: int = 64
    _cache: dict[Path, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def from_list(cls, list_path: str | Path | None, sr: int, kind: str = "noise") -> AudioBank:
        if not list_path:
            return cls(sr=sr)
        paths = read_path_list(list_path, kind=kind, suffixes=AUDIO_SUFFIXES)
        return cls(sr=sr, paths=paths, labels=relative_labels(paths))

    def __bool__(self) -> bool:
        return bool(self.paths)

    def _load(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            data, sr_file = sf.read(str(path), always_2d=False)
            self._cache[path] = resample_to(to_mono(data), sr_file, self.sr)
        return self._cache[path]

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        if not self.paths:
            raise RuntimeError("audio bank is empty")
        index = int(rng.integers(len(self.paths)))
        audio = self._load(self.paths[index])
        if len(audio) < n:
            audio = np.tile(audio, int(np.ceil(n / max(len(audio), 1))))
        start = int(rng.integers(0, max(1, len(audio) - n + 1)))
        return audio[start : start + n].copy(), self.labels[index]


__all__ = [
    "CAR_CONDITIONS",
    "HOME_CONDITIONS",
    "AudioBank",
    "scale_to_dba",
    "shaped_noise",
    "synth_noise",
]
