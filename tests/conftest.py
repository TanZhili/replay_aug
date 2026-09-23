"""Shared fixtures: a speech-like test signal with real silence around it."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

SR = 32_000


def speech_like(
    sr: int = SR,
    seconds: float = 1.2,
    lead: float = 0.3,
    tail: float = 0.2,
    seed: int = 0,
    f0: float = 140.0,
) -> np.ndarray:
    """Harmonic-plus-noise bursts with a speech-shaped spectrum, framed by silence.

    The silence matters: tests of the shared noise and of label timing need a stretch
    where no source is active yet.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    pitch = f0 * (1.0 + 0.08 * np.sin(2 * np.pi * 1.3 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / sr
    voiced = sum(np.sin(k * phase) / k for k in range(1, 40) if k * f0 < 0.45 * sr)
    voiced = voiced + 0.1 * rng.standard_normal(n)
    sos = signal.butter(2, (150.0 / (sr / 2), 5000.0 / (sr / 2)), btype="band", output="sos")
    shaped = signal.sosfilt(sos, voiced) * (0.3 + 0.7 * np.sin(2 * np.pi * 3.5 * t) ** 2)
    shaped *= np.hanning(n) ** 0.2
    shaped = 0.5 * shaped / np.max(np.abs(shaped))
    return np.concatenate([np.zeros(int(sr * lead)), shaped, np.zeros(int(sr * tail))])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def speech() -> np.ndarray:
    return speech_like()
