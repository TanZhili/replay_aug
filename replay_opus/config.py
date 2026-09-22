"""Configuration for the replay chain.

Every numeric field may be a scalar or a ``[low, high]`` range. Ranges are sampled
once per generated variant, so one config describes a whole distribution of replay
conditions rather than a single one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

Range = float | list[float]


@dataclass
class PlaybackConfig:
    enabled: bool = True
    quality: list[str] = field(default_factory=lambda: ["perfect", "high", "low"])
    quality_weights: list[float] = field(default_factory=lambda: [0.15, 0.50, 0.35])
    drive_db: Range = field(default_factory=lambda: [-9.0, -2.0])
    n_orders: int = 5
    model_dir: str | None = None


@dataclass
class RoomConfig:
    enabled: bool = True
    # measured: convolve IRs from ir_dir as-is
    # reshaped: convolve IRs from ir_dir retargeted to the sampled T60/DRR
    # synthetic: build an IR from the sampled T60/DRR
    # simulated: image-source method via pyroomacoustics
    mode: str = "synthetic"
    ir_dir: str | None = None
    t60_s: Range = field(default_factory=lambda: [0.08, 0.9])
    drr_db: Range = field(default_factory=lambda: [-3.0, 20.0])
    ir_max_ms: float = 1500.0
    room_dim_m: list[float] = field(default_factory=lambda: [3.0, 3.0, 2.4])
    room_dim_max_m: list[float] = field(default_factory=lambda: [8.0, 7.0, 3.2])
    # Loudspeaker-to-microphone distance. ASVspoof 2019 PA spans 0.1-1.5 m;
    # ASVspoof 2021 PA widened it to 0.5-2.0 m in real rooms.
    distance_m: Range = field(default_factory=lambda: [0.2, 2.0])


@dataclass
class NoiseConfig:
    enabled: bool = True
    noise_dir: str | None = None
    snr_db: Range = field(default_factory=lambda: [12.0, 40.0])


@dataclass
class CaptureConfig:
    enabled: bool = True
    mic_ir_dir: str | None = None
    agc: bool = True
    agc_target_dbfs: Range = field(default_factory=lambda: [-26.0, -16.0])
    agc_max_gain_db: float = 30.0
    adc_headroom_db: Range = field(default_factory=lambda: [-6.0, 1.0])


@dataclass
class TransportConfig:
    enabled: bool = True
    codecs: list[str | None] = field(default_factory=lambda: [None, "opus", "aac", "mp3"])
    codec_weights: list[float] = field(default_factory=lambda: [0.30, 0.40, 0.15, 0.15])
    bitrate_bps: Range = field(default_factory=lambda: [16_000, 64_000])
    clock_drift_ppm: Range = field(default_factory=lambda: [-100.0, 100.0])


@dataclass
class OutputConfig:
    sr: int = 16_000
    peak_db: Range = field(default_factory=lambda: [-3.0, -1.0])
    subtype: str = "PCM_16"
    suffix: str = ".wav"
    # How much reverberant tail to keep past the end of the input. Convolution adds
    # the full IR length, which would otherwise stretch every file by up to 1.5 s.
    tail_ms: float = 300.0


@dataclass
class Config:
    seed: int = 1234
    sr_internal: int = 48_000
    variants_per_file: int = 1
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Config:
        return _build(cls, data or {})

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.room.mode not in {"measured", "reshaped", "synthetic", "simulated"}:
            raise ValueError(f"unknown room.mode {self.room.mode!r}")
        if self.room.mode in {"measured", "reshaped"} and not self.room.ir_dir:
            raise ValueError(f"room.mode={self.room.mode!r} requires room.ir_dir")
        if len(self.playback.quality) != len(self.playback.quality_weights):
            raise ValueError("playback.quality and playback.quality_weights differ in length")
        if len(self.transport.codecs) != len(self.transport.codec_weights):
            raise ValueError("transport.codecs and transport.codec_weights differ in length")


def _build(cls, data: dict[str, Any]):
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key {key!r} in {cls.__name__}")
        target = known[key].type
        resolved = _SECTIONS.get((cls.__name__, key))
        if resolved is not None and isinstance(value, dict):
            kwargs[key] = _build(resolved, value)
        elif is_dataclass(target) and isinstance(value, dict):
            kwargs[key] = _build(target, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


# Field annotations are strings under ``from __future__ import annotations``, so the
# nested dataclass types are resolved through this table instead of the annotation.
_SECTIONS = {
    ("Config", "playback"): PlaybackConfig,
    ("Config", "room"): RoomConfig,
    ("Config", "noise"): NoiseConfig,
    ("Config", "capture"): CaptureConfig,
    ("Config", "transport"): TransportConfig,
    ("Config", "output"): OutputConfig,
}
