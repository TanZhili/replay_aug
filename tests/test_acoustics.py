"""Room model: checked against the physics it claims to follow."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from replay_opus.acoustics import (
    BANDS_HZ,
    SPEED_OF_SOUND,
    Directivity,
    Room,
    absorption_for_rt60,
    band_weights,
    image_sources,
    room_impulse_response,
    scattering_bands,
)
from replay_opus.metrics import estimate_drr, estimate_t60

SR = 32_000


def _band(h: np.ndarray, fc: float) -> np.ndarray:
    sos = signal.butter(
        3, (fc / 1.41 / (SR / 2), min(fc * 1.41 / (SR / 2), 0.99)), btype="band", output="sos"
    )
    return signal.sosfiltfilt(sos, h)


def _living_room(t60: float = 0.55) -> Room:
    dims = np.array([5.5, 4.2, 2.7])
    return Room(
        dims,
        absorption_for_rt60(dims, np.full(7, t60), np.array([1, 1, 1, 1, 1.5, 0.8])),
        scattering_bands(0.25),
    )


def test_band_weights_are_a_partition_of_unity():
    weights = band_weights(4096, SR)
    assert np.allclose(weights.sum(axis=0), 1.0)
    assert np.all(weights >= 0)


def test_free_field_gives_one_over_distance_at_the_right_time(rng):
    room = Room([20, 20, 20], np.full((6, 7), 0.999), scattering_bands(0.0))
    src, mic = np.array([10.0, 10.0, 10.0]), np.array([12.0, 10.0, 10.0])
    h, info = room_impulse_response(room, src, mic, SR, rng, jitter_m=0.0)
    assert info["delay_s"] == pytest.approx(2.0 / SPEED_OF_SOUND, abs=1e-5)
    # In-band energy of the direct path is 1/d^2.
    energy = np.sum(_band(h, 1000.0) ** 2) / np.sum(
        _band(np.eye(1, len(h), 100).ravel(), 1000.0) ** 2
    )
    assert energy == pytest.approx(0.25, rel=0.05)
    assert int(np.argmax(np.abs(h))) == pytest.approx(2.0 / SPEED_OF_SOUND * SR, abs=1)


def test_absorption_solver_hits_the_requested_rt60():
    dims = np.array([4.0, 3.5, 2.6])
    target = np.array([0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.35])
    room = Room(dims, absorption_for_rt60(dims, target, np.array([1, 1, 0.7, 0.7, 2.0, 0.5])))
    assert room.rt60() == pytest.approx(target, rel=0.01)


@pytest.mark.parametrize("fc", [500.0, 1000.0, 4000.0])
def test_generated_ir_decays_at_the_room_rt60(fc):
    room = _living_room()
    t30 = []
    for seed in range(4):
        h, _ = room_impulse_response(
            room, [1.0, 2.0, 1.2], [4.2, 2.6, 1.0], SR, np.random.default_rng(seed)
        )
        t30.append(estimate_t60(_band(h, fc), SR))
    want = room.rt60()[int(np.argmin(np.abs(BANDS_HZ - fc)))]
    assert np.median(t30) == pytest.approx(want, rel=0.12)


def test_car_cabin_decay_matches_despite_its_shape():
    dims = np.array([2.5, 1.45, 1.15])
    room = Room(
        dims,
        absorption_for_rt60(dims, np.full(7, 0.07), np.array([0.5, 1.2, 0.6, 0.6, 1.8, 1.3])),
        scattering_bands(0.4),
    )
    t30 = [
        estimate_t60(
            _band(
                room_impulse_response(
                    room, [1.15, 0.37, 0.95], [0.6, 0.72, 1.11], SR, np.random.default_rng(s)
                )[0],
                1000.0,
            ),
            SR,
        )
        for s in range(4)
    ]
    assert np.median(t30) == pytest.approx(0.07, rel=0.2)


def test_drr_follows_diffuse_field_theory():
    room = _living_room()
    src = np.array([1.0, 2.0, 1.2])
    t = room.rt60()[3]
    for d in (0.5, 1.0, 2.0):
        mic = src + np.array([d, 0.0, 0.0])
        irs = [
            room_impulse_response(room, src, mic, SR, np.random.default_rng(s))[0] for s in range(3)
        ]
        drr = np.mean([estimate_drr(h, SR) for h in irs])
        theory = 10 * np.log10(
            (1 / d**2) / (312.2 * t / room.volume * np.exp(-13.82 * d / SPEED_OF_SOUND / t))
        )
        assert drr == pytest.approx(theory, abs=2.0)


def test_directivity_changes_the_direct_path_not_the_late_field():
    room = _living_room()
    src = np.array([1.5, 2.0, 1.2])
    mic = src + np.array([2.0, 0.0, 0.0])
    alpha = np.array([0.97, 0.93, 0.86, 0.76, 0.66, 0.58, 0.51])
    toward, _ = room_impulse_response(
        room, src, mic, SR, np.random.default_rng(1), Directivity([1, 0, 0], alpha)
    )
    away, _ = room_impulse_response(
        room, src, mic, SR, np.random.default_rng(1), Directivity([-1, 0, 0], alpha)
    )
    d0 = round(2.0 / SPEED_OF_SOUND * SR)
    direct = lambda h: np.sum(_band(h, 2000.0)[d0 - 20 : d0 + 20] ** 2)  # noqa: E731
    assert 10 * np.log10(direct(toward) / direct(away)) > 6.0
    late = slice(int(0.15 * SR), None)
    assert np.sum(toward[late] ** 2) == pytest.approx(np.sum(away[late] ** 2), rel=1e-6)


def test_image_density_is_one_per_room_volume():
    room = _living_room()
    reach = 20.0
    positions, counts, parity = image_sources(
        room, np.array([1.0, 2.0, 1.2]), np.array([4.2, 2.6, 1.0]), reach
    )
    expected = 4.0 / 3.0 * np.pi * reach**3 / room.volume
    assert len(positions) == pytest.approx(expected, rel=0.1)
    assert counts.shape == (len(positions), 6) and parity.shape == (len(positions), 3)


def test_same_rng_same_ir(rng):
    room = _living_room()
    a, _ = room_impulse_response(room, [1, 1, 1], [3, 2, 1.2], SR, np.random.default_rng(7))
    b, _ = room_impulse_response(room, [1, 1, 1], [3, 2, 1.2], SR, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_sources_outside_the_room_are_rejected(rng):
    with pytest.raises(ValueError, match="outside the room"):
        room_impulse_response(_living_room(), [9.0, 1.0, 1.0], [1.0, 1.0, 1.0], SR, rng)
