"""Geometric room acoustics shared by every source in a scene.

A live talker and a TV in the same living room excite the same room: same
reverberation, same noise, same microphone. What differs is where each source sits,
which way it radiates, and what it radiates. This module turns a shoebox room, a
source position and directivity, and a microphone position into an impulse response:

* direct path and early reflections by the image-source method (Allen & Berkley,
  JASA 1979), with octave-band wall absorption, air absorption (ISO 9613-1) and a
  frequency-dependent source directivity evaluated for every image;
* a diffuse field whose level follows from the same room (Barron's revised theory).
  Its expected energy density, ``4*pi*c / (V*Q)`` per second decaying at the Eyring
  rate, is exactly the mean energy density of the image lattice.

Real rooms are furnished and car cabins are full of seats, so reflections are
partly diffuse. Each reflection keeps a share ``(1 - s)`` of its energy on the
specular path and hands the rest to the diffuse field, which therefore builds up
from the direct arrival instead of switching on at the mixing time. Without this a
bare shoebox - a car cabin especially - decays far slower along its grazing paths
than the Eyring rate says, and the early part would not meet the tail.

IRs are scaled so that a source whose on-axis free-field pressure at 1 m is ``p(t)``
produces ``(h * p)(t)`` at the microphone. Rendering in pascals this way keeps the
talker, the playback devices and the noise at physically consistent levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import fft as sp_fft

SPEED_OF_SOUND = 343.0
BANDS_HZ = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
N_BANDS = len(BANDS_HZ)
# ISO 9613-1 attenuation at 20 C and 50 % relative humidity, dB per metre.
AIR_DB_PER_M = np.array([0.00044, 0.00131, 0.00273, 0.00466, 0.00986, 0.0328, 0.117])
AIR_ENERGY_PER_M = AIR_DB_PER_M / (10.0 * np.log10(np.e))
WALLS = ("x0", "x1", "y0", "y1", "z0", "z1")
SABINE = 24.0 * np.log(10.0) / SPEED_OF_SOUND  # 0.161 s/m
DECAY_60DB = 6.0 * np.log(10.0)  # energy decays as exp(-DECAY_60DB * t / T60)

_FD_HALF = 16  # windowed-sinc fractional delay: 32 taps
_PAD_S = 0.02  # head room for the fractional-delay taps and zero-phase band filters
# Scattering rises with frequency: surfaces look rough once they are comparable to
# the wavelength.
SCATTERING_PROFILE = np.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])


def scattering_bands(s_mid: float) -> np.ndarray:
    return np.clip(s_mid * SCATTERING_PROFILE, 0.0, 0.9)


@dataclass
class Room:
    """Shoebox room. Walls are ordered x0, x1, y0, y1, z0 (floor), z1 (ceiling)."""

    dims: np.ndarray
    absorption: np.ndarray  # (6, N_BANDS) energy absorption coefficients
    scattering: np.ndarray = field(default_factory=lambda: scattering_bands(0.2))

    def __post_init__(self) -> None:
        self.dims = np.asarray(self.dims, dtype=np.float64).reshape(3)
        self.absorption = np.clip(
            np.asarray(self.absorption, dtype=np.float64).reshape(6, N_BANDS), 0.0, 0.999
        )
        self.scattering = np.clip(
            np.broadcast_to(np.asarray(self.scattering, dtype=np.float64), (N_BANDS,)), 0.0, 0.95
        ).copy()
        if np.any(self.dims <= 0):
            raise ValueError(f"room dimensions must be positive, got {self.dims}")

    @property
    def volume(self) -> float:
        return float(np.prod(self.dims))

    @property
    def wall_areas(self) -> np.ndarray:
        lx, ly, lz = self.dims
        return np.array([ly * lz, ly * lz, lx * lz, lx * lz, lx * ly, lx * ly])

    @property
    def surface(self) -> float:
        return float(self.wall_areas.sum())

    def mean_absorption(self) -> np.ndarray:
        return self.wall_areas @ self.absorption / self.surface

    def rt60(self) -> np.ndarray:
        """Eyring reverberation time per octave band, air absorption included."""
        alpha = np.clip(self.mean_absorption(), 1e-4, 0.999)
        area = -self.surface * np.log1p(-alpha) + 4.0 * AIR_ENERGY_PER_M * self.volume
        return SABINE * self.volume / area

    def mixing_time(self) -> float:
        """Perceptual mixing time ``20 V/S + 12 ms`` (Lindau et al., JAES 2012)."""
        return 0.020 * self.volume / self.surface + 0.012

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        p = np.asarray(point, dtype=np.float64)
        return bool(np.all(p >= margin) and np.all(p <= self.dims - margin))

    def describe(self) -> dict:
        return {
            "dims_m": [round(float(d), 3) for d in self.dims],
            "volume_m3": round(self.volume, 2),
            "rt60_s": [round(float(t), 3) for t in self.rt60()],
            "mean_absorption": [round(float(a), 3) for a in self.mean_absorption()],
            "scattering_1k": round(float(self.scattering[3]), 3),
        }


def absorption_for_rt60(
    dims: np.ndarray, rt60: np.ndarray, wall_weights: np.ndarray | None = None
) -> np.ndarray:
    """Per-wall, per-band absorption that gives the requested Eyring RT60.

    ``wall_weights`` sets how absorption is distributed (a carpeted floor absorbs more
    than a plastered ceiling; a car's glass absorbs less than its seats); a per-band
    scale is then solved so the area-weighted mean hits the target.
    """
    dims = np.asarray(dims, dtype=np.float64)
    rt60 = np.broadcast_to(np.asarray(rt60, dtype=np.float64), (N_BANDS,))
    weights = np.ones(6) if wall_weights is None else np.asarray(wall_weights, dtype=np.float64)
    probe = Room(dims, np.zeros((6, N_BANDS)))
    areas, surface, volume = probe.wall_areas, probe.surface, probe.volume

    area_needed = SABINE * volume / np.maximum(rt60, 1e-3) - 4.0 * AIR_ENERGY_PER_M * volume
    target = np.clip(1.0 - np.exp(-np.maximum(area_needed, 0.0) / surface), 0.01, 0.95)

    lo_alpha, hi_alpha = 0.005, 0.98
    k_lo = np.zeros(N_BANDS)
    k_hi = np.full(N_BANDS, hi_alpha / max(float(weights.min()), 1e-3))
    for _ in range(60):
        k = 0.5 * (k_lo + k_hi)
        mean = areas @ np.clip(weights[:, None] * k[None, :], lo_alpha, hi_alpha) / surface
        too_low = mean < target
        k_lo = np.where(too_low, k, k_lo)
        k_hi = np.where(too_low, k_hi, k)
    k = 0.5 * (k_lo + k_hi)
    return np.clip(weights[:, None] * k[None, :], lo_alpha, hi_alpha)


@dataclass
class Directivity:
    """Cardioid-family radiation per octave band: ``g = a + (1 - a) cos(theta)``.

    ``alpha`` of 1 is omnidirectional, 0.5 a cardioid. A human mouth and a small
    loudspeaker are both near-omni at low frequency and beam increasingly towards
    high frequency, but by different amounts and in different directions - a TV
    driver firing at the floor sends its high frequencies somewhere a talker's
    mouth does not.
    """

    axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    alpha: np.ndarray = field(default_factory=lambda: np.ones(N_BANDS))
    floor_db: float = -25.0

    def __post_init__(self) -> None:
        axis = np.asarray(self.axis, dtype=np.float64).reshape(3)
        self.axis = axis / (np.linalg.norm(axis) + 1e-12)
        self.alpha = np.clip(np.asarray(self.alpha, dtype=np.float64).reshape(N_BANDS), 0.5, 1.0)

    def gains(self, directions: np.ndarray) -> np.ndarray:
        """Amplitude gain per band for unit emission directions ``(N, 3)``."""
        cos = np.asarray(directions, dtype=np.float64) @ self.axis
        g = self.alpha[None, :] + (1.0 - self.alpha[None, :]) * cos[:, None]
        return np.maximum(g, 10.0 ** (self.floor_db / 20.0))

    def q_factor(self) -> np.ndarray:
        """On-axis directivity factor: on-axis intensity over the spherical mean."""
        return 1.0 / (self.alpha**2 + (1.0 - self.alpha) ** 2 / 3.0)


@lru_cache(maxsize=32)
def band_weights(n_fft: int, sr: int) -> np.ndarray:
    """Zero-phase octave-band weights on the rfft grid that sum to exactly one.

    Adjacent bands cross over with cos^2/sin^2 ramps on a log-frequency axis, so a
    set of per-band gains becomes a smooth frequency response and equal gains give a
    perfectly flat one.
    """
    freqs = sp_fft.rfftfreq(n_fft, 1.0 / sr)
    position = np.interp(
        np.log2(np.maximum(freqs, 1.0)), np.log2(BANDS_HZ), np.arange(N_BANDS, dtype=np.float64)
    )
    lower = np.minimum(np.floor(position).astype(int), N_BANDS - 2)
    frac = position - lower
    weights = np.zeros((N_BANDS, len(freqs)))
    columns = np.arange(len(freqs))
    weights[lower, columns] = np.cos(0.5 * np.pi * frac) ** 2
    weights[lower + 1, columns] += np.sin(0.5 * np.pi * frac) ** 2
    return weights


def _axis_images(length: float, s: float, r: float, reach: float):
    m = np.arange(
        -int(np.ceil(reach / (2.0 * length))) - 1, int(np.ceil(reach / (2.0 * length))) + 2
    )
    coords = np.concatenate([2.0 * m * length + s, 2.0 * m * length - s])
    hits_low = np.concatenate([np.abs(m), np.abs(m - 1)])
    hits_high = np.concatenate([np.abs(m), np.abs(m)])
    parity = np.concatenate([np.zeros_like(m), np.ones_like(m)])
    keep = np.abs(coords - r) <= reach
    return coords[keep], hits_low[keep], hits_high[keep], parity[keep]


def image_sources(
    room: Room, src: np.ndarray, mic: np.ndarray, reach: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All images within ``reach`` metres of the microphone.

    Returns positions ``(N, 3)``, wall-hit counts ``(N, 6)`` and mirror parity
    ``(N, 3)``. An image mirrored an odd number of times along an axis left the real
    source with that velocity component reversed, which is what the directivity
    has to be evaluated for.
    """
    axes = [_axis_images(room.dims[a], src[a], mic[a], reach) for a in range(3)]
    (x, xl, xh, xp), (y, yl, yh, yp), (z, zl, zh, zp) = axes
    dist2 = (
        (x[:, None, None] - mic[0]) ** 2
        + (y[None, :, None] - mic[1]) ** 2
        + (z[None, None, :] - mic[2]) ** 2
    )
    ix, iy, iz = np.nonzero(dist2 <= reach**2)
    positions = np.stack([x[ix], y[iy], z[iz]], axis=1)
    counts = np.stack([xl[ix], xh[ix], yl[iy], yh[iy], zl[iz], zh[iz]], axis=1)
    parity = np.stack([xp[ix], yp[iy], zp[iz]], axis=1)
    return positions, counts, parity


def _image_amplitudes(
    room: Room,
    src: np.ndarray,
    mic: np.ndarray,
    directivity: Directivity,
    reach: float,
    jitter_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arrival time, per-band amplitude and reflection order of every image in reach."""
    positions, counts, parity = image_sources(room, src, mic, reach)
    order = counts.sum(axis=1)
    reflected = order > 0
    if jitter_m > 0 and np.any(reflected):
        n_moved = int(reflected.sum())
        jitter = rng.standard_normal((n_moved, 3))
        radius = jitter_m * rng.random(n_moved) ** (1 / 3) / np.linalg.norm(jitter, axis=1)
        positions[reflected] += jitter * radius[:, None]

    vectors = mic - positions
    dist = np.maximum(np.linalg.norm(vectors, axis=1), 0.05)
    emission = vectors / dist[:, None] * (1 - 2 * parity)
    # Pressure reflection coefficient per wall, and the specular share kept per bounce.
    log_beta = 0.5 * np.log1p(-room.absorption)
    log_specular = 0.5 * np.log1p(-room.scattering)
    amplitude = (
        np.exp(counts @ log_beta + order[:, None] * log_specular[None, :])
        * 10.0 ** (-AIR_DB_PER_M[None, :] * dist[:, None] / 20.0)
        * directivity.gains(emission)
        / dist[:, None]
    )
    return dist / SPEED_OF_SOUND, amplitude, order


def _fractional_delay(delay: np.ndarray, n_fft: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter indices, Hann-windowed sinc taps and validity mask for each delay."""
    base = np.floor(delay).astype(np.int64)
    taps = np.arange(-_FD_HALF + 1, _FD_HALF + 1)
    offset = taps[None, :] - (delay - base)[:, None]
    kernel = np.sinc(offset) * 0.5 * (1.0 + np.cos(np.pi * np.clip(offset / _FD_HALF, -1.0, 1.0)))
    index = (base[:, None] + taps[None, :]).ravel()
    return index, kernel, (index >= 0) & (index < n_fft)


def room_impulse_response(
    room: Room,
    src: np.ndarray,
    mic: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    directivity: Directivity | None = None,
    early_ms: float | None = None,
    jitter_m: float = 0.05,
    max_seconds: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Impulse response from ``src`` to ``mic``, in pressure re 1 m on-axis.

    ``jitter_m`` moves every reflection's image by up to that distance, the
    randomised image method that breaks up the sweeping echoes a perfectly
    rectangular room would otherwise produce. The direct path is never moved.
    """
    src = np.asarray(src, dtype=np.float64)
    mic = np.asarray(mic, dtype=np.float64)
    if not (room.contains(src, -1e-6) and room.contains(mic, -1e-6)):
        raise ValueError(f"source {src} or microphone {mic} lies outside the room {room.dims}")
    directivity = directivity or Directivity()
    c = SPEED_OF_SOUND
    rt60 = room.rt60()
    distance = max(float(np.linalg.norm(src - mic)), 0.05)

    if early_ms is None:
        t_early = float(np.clip(2.0 * room.mixing_time(), 0.025, 0.08))
    else:
        t_early = early_ms / 1000.0
    xfade = 0.004
    t_mix = distance / c + t_early
    arrival, amplitude, order = _image_amplitudes(
        room, src, mic, directivity, c * (t_mix + xfade / 2), jitter_m, rng
    )
    ramp = np.clip((arrival - (t_mix - xfade / 2)) / xfade, 0.0, 1.0)
    amplitude *= np.cos(0.5 * np.pi * ramp)[:, None]
    # Scattered energy has been reflected at least once, so none of it can beat the
    # first reflection to the microphone.
    first_reflection = float(arrival[order == 1].min()) if np.any(order == 1) else distance / c

    t_end = float(min(max(distance / c + rt60.max() + 0.02, t_mix + 0.02), max_seconds))
    n_out = int(np.ceil(t_end * sr))
    pad = int(_PAD_S * sr)
    n_fft = sp_fft.next_fast_len(n_out + 2 * pad, real=True)
    index, kernel, valid = _fractional_delay(arrival * sr + pad, n_fft)

    t_abs = (np.arange(n_fft) - pad) / sr
    t_pos = np.maximum(t_abs, 0.0)
    takeover = np.sin(0.5 * np.pi * np.clip((t_abs - (t_mix - xfade / 2)) / xfade, 0.0, 1.0)) ** 2
    onset = np.clip((t_abs - first_reflection) / 0.001, 0.0, 1.0)
    fade_out = np.clip((t_end - t_abs) / 0.01, 0.0, 1.0)
    # Mean number of reflections undergone by energy arriving at time t (mean free path 4V/S).
    bounces = c * t_pos * room.surface / (4.0 * room.volume)
    noise = rng.standard_normal(n_fft)
    q = directivity.q_factor()

    weights = band_weights(n_fft, sr)
    spectrum = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
    for b in range(N_BANDS):
        train = np.bincount(
            index[valid], weights=(amplitude[:, b, None] * kernel).ravel()[valid], minlength=n_fft
        )
        density = (
            4.0 * np.pi * c / (room.volume * sr * q[b]) * np.exp(-DECAY_60DB * t_pos / rt60[b])
        )
        specular = (1.0 - room.scattering[b]) ** bounces
        share = (1.0 - specular) + specular * takeover
        envelope = np.sqrt(density * share) * onset * fade_out
        spectrum += weights[b] * sp_fft.rfft(train + noise * envelope)
    h = sp_fft.irfft(spectrum, n_fft)[pad : pad + n_out]

    direct_gain = directivity.gains((mic - src)[None, :] / distance)[0, 3] / distance
    return h, {
        "distance_m": round(distance, 3),
        "delay_s": round(distance / c, 5),
        "direct_gain_db": round(float(20.0 * np.log10(direct_gain)), 2),
        "n_images": len(arrival),
        "t_mix_ms": round(1000.0 * t_mix, 1),
    }
