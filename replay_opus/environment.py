"""Environment presets and placement of the microphone, talkers and device drivers.

An environment preset describes a *distribution* of scenes: room size and
reverberation, where the capturing device tends to sit, where people talk from,
which playback devices are around and how noisy it is. Home rooms use a free
furniture-style layout; the car cabin uses fixed seats and the speaker positions of
a typical door/dash/rear audio system.

Coordinates are metres inside the shoebox. In the car, x runs from the dashboard to
the rear, y from the left door to the right door, z from the floor to the roof.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .acoustics import Room, absorption_for_rt60, scattering_bands
from .dsp import choose, sample_range

_HOME_NOISE = {
    "quiet": {"weight": 0.35, "level_dba": [24.0, 35.0]},
    "hvac": {"weight": 0.25, "level_dba": [32.0, 46.0]},
    "appliance": {"weight": 0.2, "level_dba": [30.0, 44.0]},
    "street": {"weight": 0.2, "level_dba": [30.0, 48.0]},
}

ENVIRONMENTS: dict[str, dict] = {
    "living_room": {
        "layout": "room",
        "family": "home",
        "dims_min_m": [3.8, 3.2, 2.5],
        "dims_max_m": [7.5, 6.0, 3.1],
        "rt60_s": [0.30, 0.70],
        "rt60_shape": [1.30, 1.15, 1.00, 1.00, 0.95, 0.85, 0.70],
        "wall_weights": [1.0, 1.0, 1.0, 1.0, 1.5, 0.8],
        "scattering": [0.15, 0.35],
        "mic_height_m": [0.4, 1.5],
        "mic_near_wall_prob": 0.6,
        "talker_distance_m": [0.5, 5.0],
        "talker_height_m": [1.0, 1.75],
        "face_mic_prob": 0.7,
        "devices": {
            "tv": 0.38,
            "soundbar": 0.08,
            "phone": 0.18,
            "tablet": 0.1,
            "laptop": 0.08,
            "smart_speaker": 0.08,
            "bluetooth_speaker": 0.1,
        },
        "noise": copy.deepcopy(_HOME_NOISE),
    },
    "bedroom": {
        "layout": "room",
        "family": "home",
        "dims_min_m": [3.0, 2.8, 2.4],
        "dims_max_m": [5.2, 4.5, 2.9],
        "rt60_s": [0.25, 0.55],
        "rt60_shape": [1.35, 1.2, 1.0, 0.95, 0.9, 0.8, 0.65],
        "wall_weights": [1.0, 1.0, 1.0, 1.0, 1.6, 0.8],
        "scattering": [0.2, 0.4],
        "mic_height_m": [0.4, 1.3],
        "mic_near_wall_prob": 0.7,
        "talker_distance_m": [0.4, 3.5],
        "talker_height_m": [0.6, 1.7],
        "face_mic_prob": 0.6,
        "devices": {
            "tv": 0.25,
            "phone": 0.32,
            "tablet": 0.15,
            "laptop": 0.12,
            "smart_speaker": 0.1,
            "bluetooth_speaker": 0.06,
        },
        "noise": {
            "quiet": {"weight": 0.5, "level_dba": [20.0, 32.0]},
            "hvac": {"weight": 0.3, "level_dba": [28.0, 40.0]},
            "street": {"weight": 0.2, "level_dba": [26.0, 42.0]},
        },
    },
    "kitchen": {
        "layout": "room",
        "family": "home",
        "dims_min_m": [2.4, 2.0, 2.4],
        "dims_max_m": [5.0, 4.0, 3.0],
        "rt60_s": [0.40, 1.00],
        "rt60_shape": [1.15, 1.1, 1.0, 1.0, 0.95, 0.9, 0.8],
        "wall_weights": [0.8, 0.8, 0.8, 0.8, 0.9, 0.8],
        "scattering": [0.1, 0.3],
        "mic_height_m": [0.8, 1.6],
        "mic_near_wall_prob": 0.8,
        "talker_distance_m": [0.4, 3.5],
        "talker_height_m": [1.4, 1.75],
        "face_mic_prob": 0.5,
        "devices": {
            "phone": 0.35,
            "tablet": 0.2,
            "radio": 0.15,
            "smart_speaker": 0.15,
            "bluetooth_speaker": 0.1,
            "tv": 0.05,
        },
        "noise": {
            "quiet": {"weight": 0.2, "level_dba": [26.0, 36.0]},
            "appliance": {"weight": 0.5, "level_dba": [34.0, 54.0]},
            "hvac": {"weight": 0.1, "level_dba": [32.0, 46.0]},
            "street": {"weight": 0.2, "level_dba": [30.0, 46.0]},
        },
    },
    "study": {
        "layout": "room",
        "family": "home",
        "dims_min_m": [2.8, 2.6, 2.4],
        "dims_max_m": [5.0, 4.5, 3.0],
        "rt60_s": [0.30, 0.65],
        "rt60_shape": [1.25, 1.1, 1.0, 1.0, 0.95, 0.85, 0.7],
        "wall_weights": [1.0, 1.0, 1.0, 1.0, 1.3, 0.8],
        "scattering": [0.15, 0.35],
        "mic_height_m": [0.7, 1.4],
        "mic_near_wall_prob": 0.6,
        "talker_distance_m": [0.4, 3.0],
        "talker_height_m": [1.0, 1.75],
        "face_mic_prob": 0.7,
        "devices": {
            "laptop": 0.35,
            "phone": 0.25,
            "tablet": 0.1,
            "smart_speaker": 0.15,
            "bluetooth_speaker": 0.15,
        },
        "noise": {
            "quiet": {"weight": 0.4, "level_dba": [24.0, 35.0]},
            "hvac": {"weight": 0.4, "level_dba": [30.0, 44.0]},
            "street": {"weight": 0.2, "level_dba": [28.0, 44.0]},
        },
    },
    "car_cabin": {
        "layout": "car",
        "family": "car",
        # A shoebox stand-in for the cabin air volume of a compact-to-midsize car.
        "dims_min_m": [2.1, 1.35, 1.05],
        "dims_max_m": [2.9, 1.6, 1.25],
        "rt60_s": [0.04, 0.12],
        "rt60_shape": [1.5, 1.25, 1.0, 0.9, 0.85, 0.8, 0.75],
        # windshield, rear window/seats, side glass+doors, floor+seats, headliner
        "wall_weights": [0.5, 1.2, 0.6, 0.6, 1.8, 1.3],
        "scattering": [0.3, 0.5],
        "drive_side": "left",
        "mic_positions": {"overhead": 0.7, "mirror": 0.3},
        "seats": {"driver": 0.55, "front_passenger": 0.25, "rear": 0.2},
        "face_mic_prob": 0.25,
        "devices": {"car_audio": 0.7, "phone": 0.3},
        "noise": {
            "idle": {"weight": 0.2, "level_dba": [36.0, 48.0]},
            "city": {"weight": 0.4, "level_dba": [52.0, 64.0]},
            "highway": {"weight": 0.4, "level_dba": [62.0, 74.0]},
        },
    },
}


@dataclass
class Pose:
    position: np.ndarray
    axis: np.ndarray
    role: str = "full"  # woofer / tweeter drivers of a multi-way system get crossovers

    def describe(self) -> dict:
        return {
            "pos_m": [round(float(v), 3) for v in self.position],
            "axis": [round(float(v), 3) for v in self.axis],
            "role": self.role,
        }


@dataclass
class Placement:
    """Where a source sits: one pose for a talker, one per driver for a device."""

    poses: list[Pose]
    meta: dict = field(default_factory=dict)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def _direction(yaw: float, pitch: float = 0.0) -> np.ndarray:
    return np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])


def _clamp(p, dims, margin: float = 0.02) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), margin, np.asarray(dims) - margin)


def _deg(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.deg2rad(rng.uniform(lo, hi)))


def build_room(preset: dict, rng: np.random.Generator) -> Room:
    lo = np.asarray(preset["dims_min_m"], dtype=np.float64)
    hi = np.asarray(preset["dims_max_m"], dtype=np.float64)
    dims = lo + rng.random(3) * (hi - lo)
    rt_mid = sample_range(rng, preset["rt60_s"])
    shape = np.asarray(preset["rt60_shape"], dtype=np.float64) * rng.uniform(0.92, 1.08, 7)
    weights = np.asarray(preset["wall_weights"], dtype=np.float64) * rng.uniform(0.8, 1.2, 6)
    absorption = absorption_for_rt60(dims, rt_mid * shape, weights)
    return Room(dims, absorption, scattering_bands(sample_range(rng, preset["scattering"])))


# ---------------------------------------------------------------------------
# Microphone
# ---------------------------------------------------------------------------


def place_mic(preset: dict, dims: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    if preset["layout"] == "car":
        spot = choose(rng, preset["mic_positions"])
        _, width, height = dims
        if spot == "overhead":
            p = [
                rng.uniform(0.45, 0.75),
                width / 2 + rng.uniform(-0.05, 0.05),
                height - rng.uniform(0.03, 0.06),
            ]
        else:
            p = [
                rng.uniform(0.3, 0.45),
                width / 2 + rng.uniform(-0.04, 0.04),
                height - rng.uniform(0.06, 0.12),
            ]
        return _clamp(p, dims), {"mic_spot": spot}

    z = sample_range(rng, preset["mic_height_m"])
    if rng.random() < preset["mic_near_wall_prob"]:
        wall = choose(rng, {"x0": dims[1], "x1": dims[1], "y0": dims[0], "y1": dims[0]})
        off = rng.uniform(0.05, 0.6)
        if wall[0] == "x":
            along = rng.uniform(0.3, dims[1] - 0.3)
            x = off if wall == "x0" else dims[0] - off
            p = [x, along, z]
        else:
            along = rng.uniform(0.3, dims[0] - 0.3)
            y = off if wall == "y0" else dims[1] - off
            p = [along, y, z]
        return _clamp(p, dims), {"mic_spot": f"wall_{wall}"}
    p = [rng.uniform(0.5, dims[0] - 0.5), rng.uniform(0.5, dims[1] - 0.5), z]
    return _clamp(p, dims), {"mic_spot": "free"}


# ---------------------------------------------------------------------------
# Talkers
# ---------------------------------------------------------------------------


def _place_at_distance(
    dims: np.ndarray,
    mic: np.ndarray,
    distance_range,
    height_range,
    rng: np.random.Generator,
    margin: float = 0.25,
) -> np.ndarray:
    lo, hi = float(distance_range[0]), float(distance_range[1])
    for attempt in range(400):
        if attempt and attempt % 100 == 0:
            hi = max(lo * 1.05, hi * 0.75)
        d = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        z = sample_range(rng, height_range)
        horizontal = np.sqrt(max(d * d - (z - mic[2]) ** 2, 0.04))
        yaw = rng.uniform(0.0, 2.0 * np.pi)
        p = np.array([mic[0] + horizontal * np.cos(yaw), mic[1] + horizontal * np.sin(yaw), z])
        if np.all(p >= margin) and np.all(p <= dims - margin):
            return p
    return _clamp(dims / 2 + (mic - dims / 2) * -0.5, dims, margin)


def place_talker(
    preset: dict, dims: np.ndarray, mic: np.ndarray, rng: np.random.Generator
) -> Placement:
    face_prob = float(preset.get("face_mic_prob", 0.6))
    if preset["layout"] == "car":
        seat = choose(rng, preset["seats"])
        length, width, height = dims
        side = -1.0 if preset.get("drive_side", "left") == "left" else 1.0
        if seat == "driver":
            p = [
                rng.uniform(1.0, 1.3),
                width / 2 + side * rng.uniform(0.30, 0.40),
                rng.uniform(0.85, 1.0),
            ]
        elif seat == "front_passenger":
            p = [
                rng.uniform(1.0, 1.3),
                width / 2 - side * rng.uniform(0.30, 0.40),
                rng.uniform(0.85, 1.0),
            ]
        else:
            lane = float(rng.choice([-0.35, 0.0, 0.35]))
            p = [
                rng.uniform(1.75, min(2.2, length - 0.3)),
                width / 2 + lane + rng.uniform(-0.05, 0.05),
                rng.uniform(0.8, 0.95),
            ]
        p = _clamp(p, dims, 0.1)
        p[2] = min(p[2], height - 0.12)
        if rng.random() < face_prob:
            to_mic = mic - p
            yaw = np.arctan2(to_mic[1], to_mic[0]) + _deg(rng, -20, 20)
        else:
            yaw = np.pi + _deg(rng, -25, 25)  # facing the windshield
        axis = _direction(yaw, _deg(rng, -10, 15))
        return Placement([Pose(p, axis)], {"seat": seat})

    p = _place_at_distance(dims, mic, preset["talker_distance_m"], preset["talker_height_m"], rng)
    if rng.random() < face_prob:
        to_mic = mic - p
        yaw = np.arctan2(to_mic[1], to_mic[0]) + _deg(rng, -30, 30)
        facing = "towards_mic"
    else:
        yaw = rng.uniform(0.0, 2.0 * np.pi)
        facing = "random"
    return Placement([Pose(p, _direction(yaw, _deg(rng, -10, 10)))], {"facing": facing})


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------


def _wall_frame(wall: str, dims: np.ndarray):
    """Origin corner, along-wall unit vector, into-room normal and wall length."""
    if wall == "x0":
        return (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            dims[1],
        )
    if wall == "x1":
        return (
            np.array([dims[0], 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            dims[1],
        )
    if wall == "y0":
        return (
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            dims[0],
        )
    return (
        np.array([0.0, dims[1], 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        dims[0],
    )


def _random_wall(dims: np.ndarray, rng: np.random.Generator) -> str:
    return choose(rng, {"x0": dims[1], "x1": dims[1], "y0": dims[0], "y1": dims[0]})


def _spread(centre, along, spacing: float, n: int) -> list[np.ndarray]:
    if n == 1:
        return [np.asarray(centre, dtype=np.float64)]
    offsets = np.linspace(-spacing / 2, spacing / 2, n)
    return [np.asarray(centre) + o * np.asarray(along) for o in offsets]


def _tv(dims, mic, rng):
    wall = _random_wall(dims, rng)
    origin, along, normal, length = _wall_frame(wall, dims)
    width = rng.uniform(0.95, 1.65)
    centre_along = rng.uniform(0.3, 0.7) * length
    centre_z = rng.uniform(0.9, 1.4)
    bottom = centre_z - 0.28 * width + rng.uniform(0.0, 0.05)
    base = origin + along * centre_along + normal * rng.uniform(0.04, 0.12)
    base[2] = bottom
    down_firing = rng.random() < 0.6
    axis = (
        _unit(np.array([0, 0, -1.0]) + 0.35 * normal)
        if down_firing
        else _unit(normal + np.array([0.0, 0.0, -0.15]))
    )
    spacing = rng.uniform(0.6, 0.9) * width
    poses = [Pose(_clamp(p, dims), axis) for p in _spread(base, along, spacing, 2)]
    return poses, {
        "wall": wall,
        "tv_width_m": round(width, 2),
        "firing": "down" if down_firing else "front",
    }


def _soundbar(dims, mic, rng):
    wall = _random_wall(dims, rng)
    origin, along, normal, length = _wall_frame(wall, dims)
    base = origin + along * rng.uniform(0.3, 0.7) * length + normal * rng.uniform(0.05, 0.3)
    base[2] = rng.uniform(0.5, 0.9)
    n = int(rng.choice([2, 3]))
    axis = _unit(normal + np.array([0.0, 0.0, 0.05]))
    poses = [Pose(_clamp(p, dims), axis) for p in _spread(base, along, rng.uniform(0.7, 1.0), n)]
    return poses, {"wall": wall, "channels": n}


def _handheld(dims, mic, rng, height, n_drivers, spacing, pitch_deg):
    p = _place_at_distance(dims, mic, (0.3, 4.0), height, rng, margin=0.15)
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    axis = _direction(yaw, _deg(rng, *pitch_deg))
    along = _direction(yaw + np.pi / 2)
    return [Pose(_clamp(q, dims), axis) for q in _spread(p, along, spacing, n_drivers)]


def _phone(dims, mic, rng):
    n = 1 if rng.random() < 0.45 else 2
    poses = _handheld(dims, mic, rng, (0.4, 1.3), n, rng.uniform(0.13, 0.16), (-20, 70))
    return poses, {"drivers": n}


def _tablet(dims, mic, rng):
    return _handheld(dims, mic, rng, (0.4, 1.2), 2, rng.uniform(0.15, 0.25), (-10, 60)), {}


def _laptop(dims, mic, rng):
    p = _place_at_distance(dims, mic, (0.4, 4.0), (0.65, 0.85), rng, margin=0.2)
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    style = choose(rng, {"up": 0.3, "front": 0.4, "down": 0.3})
    pitch = {"up": 80.0, "front": 5.0, "down": -45.0}[style]
    axis = _direction(yaw, np.deg2rad(pitch))
    poses = [
        Pose(_clamp(q, dims), axis)
        for q in _spread(p, _direction(yaw + np.pi / 2), rng.uniform(0.24, 0.32), 2)
    ]
    return poses, {"firing": style}


def _near_wall_box(dims, mic, rng, height, n_choices, spacing, up_prob):
    wall = _random_wall(dims, rng)
    origin, along, normal, length = _wall_frame(wall, dims)
    base = origin + along * rng.uniform(0.2, 0.8) * length + normal * rng.uniform(0.05, 0.5)
    base[2] = sample_range(rng, height)
    n = int(rng.choice(n_choices))
    axis = (
        np.array([0.0, 0.0, 1.0])
        if rng.random() < up_prob
        else _unit(normal + rng.normal(0, 0.2, 3))
    )
    return [Pose(_clamp(q, dims), axis) for q in _spread(base, along, spacing, n)], {"wall": wall}


def _smart_speaker(dims, mic, rng):
    return _near_wall_box(dims, mic, rng, (0.4, 1.5), [1], 0.0, 0.6)


def _bluetooth_speaker(dims, mic, rng):
    p = _place_at_distance(dims, mic, (0.4, 4.0), (0.3, 1.5), rng, margin=0.15)
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    n = int(rng.choice([1, 2]))
    axis = _direction(yaw, _deg(rng, -5, 20))
    poses = [
        Pose(_clamp(q, dims), axis)
        for q in _spread(p, _direction(yaw + np.pi / 2), rng.uniform(0.08, 0.18), n)
    ]
    return poses, {"drivers": n}


def _radio(dims, mic, rng):
    return _near_wall_box(dims, mic, rng, (0.8, 1.6), [1, 2], rng.uniform(0.15, 0.3), 0.0)


def _car_audio(dims, mic, rng):
    length, width, height = dims
    poses = []
    x_front = rng.uniform(0.85, 1.15)
    z_door = rng.uniform(0.25, 0.45)
    tilt = np.deg2rad(15.0)
    for y, inward in ((0.03, 1.0), (width - 0.03, -1.0)):
        poses.append(
            Pose(
                _clamp([x_front, y, z_door], dims),
                np.array([0.0, inward * np.cos(tilt), np.sin(tilt)]),
                "woofer",
            )
        )
    x_tw, y_tw, z_tw = (
        rng.uniform(0.2, 0.45),
        rng.uniform(0.08, 0.15),
        min(rng.uniform(0.75, 0.9), height - 0.2),
    )
    for y, inward in ((y_tw, 1.0), (width - y_tw, -1.0)):
        poses.append(
            Pose(_clamp([x_tw, y, z_tw], dims), _unit([0.5, inward * 0.6, 0.6]), "tweeter")
        )
    x_rear = min(rng.uniform(1.8, 2.15), length - 0.15)
    z_rear = rng.uniform(0.3, 0.5)
    for y, inward in ((0.03, 1.0), (width - 0.03, -1.0)):
        poses.append(
            Pose(
                _clamp([x_rear, y, z_rear], dims),
                np.array([0.0, inward * np.cos(tilt), np.sin(tilt)]),
            )
        )
    centre = rng.random() < 0.4
    if centre:
        poses.append(
            Pose(
                _clamp(
                    [rng.uniform(0.15, 0.3), width / 2, min(rng.uniform(0.8, 0.9), height - 0.2)],
                    dims,
                ),
                _unit([0.3, 0.0, 1.0]),
            )
        )
    return poses, {"centre_speaker": centre, "drivers": len(poses)}


def _car_phone(dims, mic, rng):
    _, width, height = dims
    spot = choose(rng, {"dash_mount": 0.5, "cupholder": 0.25, "passenger_hand": 0.25})
    if spot == "dash_mount":
        p = [rng.uniform(0.3, 0.55), width / 2 + rng.uniform(-0.35, 0.35), rng.uniform(0.75, 0.9)]
    elif spot == "cupholder":
        p = [rng.uniform(0.85, 1.15), width / 2 + rng.uniform(-0.1, 0.1), rng.uniform(0.35, 0.5)]
    else:
        p = [rng.uniform(0.9, 1.2), width / 2 + rng.uniform(0.2, 0.45), rng.uniform(0.6, 0.85)]
    p = _clamp(p, dims, 0.05)
    p[2] = min(p[2], height - 0.1)
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    axis = _direction(yaw, _deg(rng, -20, 70))
    n = 1 if rng.random() < 0.45 else 2
    poses = [
        Pose(_clamp(q, dims), axis)
        for q in _spread(p, _direction(yaw + np.pi / 2), rng.uniform(0.13, 0.16), n)
    ]
    return poses, {"spot": spot, "drivers": n}


ROOM_DEVICE_LAYOUTS = {
    "tv": _tv,
    "soundbar": _soundbar,
    "phone": _phone,
    "tablet": _tablet,
    "laptop": _laptop,
    "smart_speaker": _smart_speaker,
    "bluetooth_speaker": _bluetooth_speaker,
    "radio": _radio,
}
CAR_DEVICE_LAYOUTS = {"car_audio": _car_audio, "phone": _car_phone, "tablet": _car_phone}


def place_device(
    kind: str, preset: dict, dims: np.ndarray, mic: np.ndarray, rng: np.random.Generator
) -> Placement:
    layouts = CAR_DEVICE_LAYOUTS if preset["layout"] == "car" else ROOM_DEVICE_LAYOUTS
    if kind not in layouts:
        raise ValueError(f"device {kind!r} has no placement in a {preset['layout']!r} layout")
    for _ in range(20):
        poses, meta = layouts[kind](dims, mic, rng)
        if min(np.linalg.norm(p.position - mic) for p in poses) >= 0.25:
            break
    return Placement(poses, meta)


__all__ = [
    "ENVIRONMENTS",
    "Placement",
    "Pose",
    "build_room",
    "place_device",
    "place_mic",
    "place_talker",
]
