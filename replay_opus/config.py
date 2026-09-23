"""Configuration: YAML in, nested dataclasses out.

Numeric fields may be a scalar or a ``[low, high]`` range sampled per scene or per
source, so one config describes a distribution of conditions. The environment,
device and content *presets* live in code (``environment.py``, ``speaker.py``,
``content.py``); the YAML only needs to hold what you change - any preset key can
be overridden, and new presets added, through the ``*_presets`` / ``presets``
sections, which are deep-merged over the built-ins.
"""

from __future__ import annotations

import copy
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .content import CONTENT_PROFILES
from .environment import CAR_DEVICE_LAYOUTS, ENVIRONMENTS, ROOM_DEVICE_LAYOUTS
from .speaker import DEVICE_PRESETS

Range = float | list[float]


@dataclass
class TalkerConfig:
    level_dba: Range = field(default_factory=lambda: [55.0, 66.0])  # active speech level at 1 m
    lombard_onset_dba: float = 45.0
    lombard_slope: Range = field(default_factory=lambda: [0.2, 0.5])  # dB per dB(A) of noise
    lombard_max_db: float = 12.0
    effort_tilt_db_per_db: Range = field(default_factory=lambda: [0.2, 0.4])


@dataclass
class DeviceConfig:
    weights: dict[str, float] | None = None  # replaces every environment's device mix
    volume_slope: Range = field(default_factory=lambda: [0.0, 0.4])  # dB per dB(A) of noise
    measured: dict[str, str] = field(default_factory=dict)  # device kind -> response list
    measured_prob: float = 0.5
    presets: dict[str, dict] = field(default_factory=dict)


@dataclass
class ContentConfig:
    bed_list: str | None = None  # music / effects recordings mixed under programmes
    weights: dict[str, float] | None = None  # replaces every device's content mix
    profiles: dict[str, dict] = field(default_factory=dict)


@dataclass
class RoomConfig:
    measured: dict[str, str] = field(default_factory=dict)  # environment -> IR list
    jitter_m: float = 0.05
    early_ms: float | None = None


@dataclass
class NoiseConfig:
    enabled: bool = True
    lists: dict[str, str] = field(default_factory=dict)  # family (home / car) -> list


@dataclass
class CaptureConfig:
    fullscale_spl_db: Range = field(default_factory=lambda: [92.0, 112.0])
    self_noise_dba: Range = field(default_factory=lambda: [22.0, 34.0])
    mic_ir_list: str | None = None
    highpass_prob: float = 0.7
    highpass_hz: Range = field(default_factory=lambda: [60.0, 150.0])
    agc_prob: float = 0.4
    agc_target_dbfs: Range = field(default_factory=lambda: [-28.0, -18.0])
    agc_max_gain_db: float = 24.0
    uplink_prob: float = 0.0
    uplink_codecs: dict[str, float] = field(default_factory=lambda: {"opus": 1.0})
    uplink_bitrate_kbps: Range = field(default_factory=lambda: [16.0, 32.0])


@dataclass
class RenderConfig:
    mode: str = "paired"  # paired | mixed | live | device
    device_prob: float = 0.5  # share of device renders in mixed mode
    tail_ms: float = 300.0


@dataclass
class SceneConfig:
    duration_s: Range = field(default_factory=lambda: [10.0, 24.0])
    types: dict[str, float] = field(
        default_factory=lambda: {"both": 0.5, "device_only": 0.25, "live_only": 0.2, "empty": 0.05}
    )
    live_utterances: list[int] = field(default_factory=lambda: [1, 3])
    live_min_gap_s: float = 0.4
    device_gap_s: Range = field(default_factory=lambda: [0.15, 1.2])
    device_continuous_prob: float = 0.7
    device_voices: list[int] = field(default_factory=lambda: [1, 3])
    second_device_prob: float = 0.1
    save_stems: bool = False


@dataclass
class OutputConfig:
    sr: int = 16_000
    subtype: str = "PCM_16"
    format: str = "wav"  # wav | flac (lossless, roughly half the disk space)

    @property
    def suffix(self) -> str:
        return f".{self.format}"


@dataclass
class Config:
    seed: int = 1234
    sr_internal: int = 32_000
    variants_per_file: int = 1
    environments: dict[str, float] = field(
        default_factory=lambda: {
            "living_room": 0.3,
            "bedroom": 0.15,
            "kitchen": 0.1,
            "study": 0.05,
            "car_cabin": 0.4,
        }
    )
    environment_presets: dict[str, dict] = field(default_factory=dict)
    talker: TalkerConfig = field(default_factory=TalkerConfig)
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Config:
        return _build(cls, data or {})

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as handle:
            config = cls.from_dict(yaml.safe_load(handle))
        config.resolve_paths(Path(path).parent)
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_paths(self, base: Path) -> None:
        """Make list paths in a config file relative to the file, not to the shell."""

        def fix(value: str | None) -> str | None:
            if not value:
                return value
            p = Path(value)
            return str(p if p.is_absolute() else (base / p))

        self.content.bed_list = fix(self.content.bed_list)
        self.capture.mic_ir_list = fix(self.capture.mic_ir_list)
        for mapping in (self.room.measured, self.noise.lists, self.devices.measured):
            for key, value in list(mapping.items()):
                mapping[key] = fix(value)

    # -- presets ---------------------------------------------------------------

    def environment_table(self) -> dict[str, dict]:
        table = deep_merge(ENVIRONMENTS, self.environment_presets)
        if self.devices.weights is not None:
            for preset in table.values():
                layouts = CAR_DEVICE_LAYOUTS if preset["layout"] == "car" else ROOM_DEVICE_LAYOUTS
                preset["devices"] = {k: w for k, w in self.devices.weights.items() if k in layouts}
        return table

    def device_table(self) -> dict[str, dict]:
        table = deep_merge(DEVICE_PRESETS, self.devices.presets)
        if self.content.weights is not None:
            for preset in table.values():
                preset["content"] = dict(self.content.weights)
        return table

    def content_table(self) -> dict[str, dict]:
        return deep_merge(CONTENT_PROFILES, self.content.profiles)

    def validate(self) -> None:
        if self.render.mode not in {"paired", "mixed", "live", "device"}:
            raise ValueError(
                f"render.mode must be paired, mixed, live or device, got {self.render.mode!r}"
            )
        if self.output.sr > self.sr_internal:
            raise ValueError("output.sr cannot exceed sr_internal")
        if self.output.format not in {"wav", "flac"}:
            raise ValueError(f"output.format must be wav or flac, got {self.output.format!r}")
        envs = self.environment_table()
        for name, weight in self.environments.items():
            if name not in envs:
                raise ValueError(f"unknown environment {name!r}; known: {sorted(envs)}")
            if weight < 0:
                raise ValueError(f"negative weight for environment {name!r}")
        if not any(w > 0 for w in self.environments.values()):
            raise ValueError("environments needs at least one positive weight")
        unknown_types = set(self.scene.types) - {"both", "device_only", "live_only", "empty"}
        if unknown_types:
            raise ValueError(f"unknown scene type(s) {sorted(unknown_types)}")
        _check_preset_links(envs, self.device_table(), self.content_table())


def _check_preset_links(envs: dict, devices: dict, contents: dict) -> None:
    """Every device an environment names must exist and be placeable there, and every
    content profile a device names must exist."""
    for name, preset in envs.items():
        layouts = CAR_DEVICE_LAYOUTS if preset["layout"] == "car" else ROOM_DEVICE_LAYOUTS
        for kind, weight in preset["devices"].items():
            if weight > 0 and (kind not in devices or kind not in layouts):
                raise ValueError(
                    f"environment {name!r} lists device {kind!r}, which has no preset or placement"
                )
    for kind, preset in devices.items():
        for profile in preset["content"]:
            if profile not in contents:
                raise ValueError(f"device {kind!r} lists unknown content profile {profile!r}")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build(cls, data: dict[str, Any]):
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key {key!r} in {cls.__name__}")
        target = hints.get(key)
        if is_dataclass(target) and isinstance(value, dict):
            kwargs[key] = _build(target, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


__all__ = ["Config", "deep_merge"]
