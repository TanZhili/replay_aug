"""Additive noise stage: room noise floor and microphone self-noise.

Real recordings are never silent between words. Prefer a directory of real "silence"
recordings (BUT ReverbDB ships them alongside its RIRs, MUSAN and DEMAND also work);
the synthetic colours here are a fallback so the pipeline runs with no downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .dsp import EPS, resample_to, rms, to_mono

NOISE_SUFFIXES = (".wav", ".flac", ".ogg")
SYNTHETIC_COLOURS = ("white", "pink", "brown", "hvac")


def synth_noise(colour: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a noise floor of the requested spectral colour."""
    white = rng.standard_normal(n + sr)
    if colour == "white":
        out = white
    elif colour in ("pink", "brown"):
        # 1/f and 1/f^2 shaping via a one-pole cascade on white noise.
        poles = 1 if colour == "pink" else 2
        out = white
        for _ in range(poles):
            out = signal.lfilter([1.0], [1.0, -0.98], out)
    elif colour == "hvac":
        # Low-frequency rumble plus mains hum, the signature of an occupied room.
        out = signal.lfilter([1.0], [1.0, -0.995], white)
        t = np.arange(len(out)) / sr
        hum = 50.0 if rng.random() < 0.5 else 60.0
        out = out / (np.std(out) + EPS)
        out += 0.25 * np.sin(2 * np.pi * hum * t) + 0.1 * np.sin(2 * np.pi * 2 * hum * t)
    else:
        raise ValueError(f"unknown noise colour {colour!r}; choose from {SYNTHETIC_COLOURS}")
    out = out[sr : sr + n] if len(out) >= n + sr else out[:n]
    return out / (np.std(out) + EPS)


@dataclass
class NoiseBank:
    """Real noise recordings if available, synthetic colours otherwise."""

    root: Path | None
    sr: int
    paths: list[Path] = field(default_factory=list)
    colours: tuple[str, ...] = SYNTHETIC_COLOURS
    cache_size: int = 64
    _cache: dict[Path, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dir(cls, root: str | Path | None, sr: int) -> NoiseBank:
        if root is None:
            return cls(root=None, sr=sr)
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"noise directory not found: {root_path}")
        paths = sorted(p for p in root_path.rglob("*") if p.suffix.lower() in NOISE_SUFFIXES)
        if not paths:
            raise FileNotFoundError(f"no noise files under {root_path}")
        return cls(root=root_path, sr=sr, paths=paths)

    def _load(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            data, sr_file = sf.read(str(path), always_2d=False)
            self._cache[path] = resample_to(to_mono(data), sr_file, self.sr)
        return self._cache[path]

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        if not self.paths:
            colour = self.colours[int(rng.integers(len(self.colours)))]
            return synth_noise(colour, n, self.sr, rng), f"synthetic:{colour}"

        path = self.paths[int(rng.integers(len(self.paths)))]
        noise = self._load(path)
        if len(noise) < n:
            reps = int(np.ceil(n / max(len(noise), 1)))
            noise = np.tile(noise, reps)
        start = int(rng.integers(0, max(1, len(noise) - n + 1)))
        segment = noise[start : start + n]
        label = str(path.relative_to(self.root)) if self.root else str(path)
        return segment / (np.std(segment) + EPS), label


def mix_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise scaled so the mixture has the requested speech-to-noise ratio."""
    n = min(len(x), len(noise))
    speech_rms = rms(x[:n])
    noise_rms = rms(noise[:n])
    if noise_rms < EPS or speech_rms < EPS:
        return x
    scale = speech_rms / (noise_rms * 10.0 ** (snr_db / 20.0))
    out = x.copy()
    out[:n] = out[:n] + noise[:n] * scale
    return out
