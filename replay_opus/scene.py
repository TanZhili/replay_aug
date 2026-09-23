"""Scenes: one room, one microphone, one noise, any number of sources.

A scene fixes everything that the classes must share - room, microphone position,
background noise, capture chain - and sources are rendered into it independently:
a live talker as a human mouth at a human level, a playback device as a programme
chain driving loudspeakers somewhere else in the same room. Each part of a scene
draws from its own seeded random stream, so the noise is the same realisation
whichever source is rendered, and a scene can be rebuilt from its seed alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .acoustics import SPEED_OF_SOUND, Directivity, Room, room_impulse_response
from .capture import Capture, capture, sample_capture
from .codecs import available_codecs
from .config import Config
from .content import apply_content, sample_content
from .dsp import P_REF, active_level_db, choose, sample_range
from .environment import Pose, build_room, place_device, place_mic, place_talker
from .irbank import IRBank, scale_for_distance
from .noise import AudioBank, scale_to_dba, synth_noise
from .pathlist import read_path_list
from .speaker import device_emission, load_measured_response, sample_device
from .talker import sample_talker, talker_emission

_STREAMS = {
    "room": 1,
    "mic": 2,
    "noise": 3,
    "capture": 4,
    "capture_noise": 5,
    "talker": 6,
    "device": 7,
    "content": 8,
    "mode": 9,
    "compose": 10,
}


def derive_seed(base_seed: int, *parts) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in (base_seed, *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def stream_rng(seed: int, name: str, index: int = 0) -> np.random.Generator:
    return np.random.default_rng([int(seed), _STREAMS[name], int(index)])


def spl_dba(x_pa: np.ndarray, sr: int) -> float:
    """Active A-weighted level of a pressure signal, in dB SPL."""
    return active_level_db(x_pa, sr) - 20.0 * np.log10(P_REF)


@dataclass
class Scene:
    key: str
    seed: int
    environment: str
    preset: dict
    room: Room
    mic: np.ndarray
    mic_meta: dict
    noise_condition: str
    noise_dba: float
    capture: Capture
    measured: dict | None = None
    used_irs: list[int] = field(default_factory=list)

    def describe(self) -> dict:
        out = {
            "environment": self.environment,
            "family": self.preset["family"],
            "room": self.room.describe(),
            "mic_pos_m": [round(float(v), 3) for v in self.mic],
            **self.mic_meta,
            "noise_condition": self.noise_condition,
            "noise_dba": round(self.noise_dba, 2),
            "capture": self.capture.describe(),
        }
        if self.measured is not None:
            out["measured_room"] = self.measured["room"]
        return out


@dataclass
class SourceRender:
    role: str  # "live" or "device"
    signal: np.ndarray  # pressure at the microphone, pascals
    delay_s: float  # earliest direct-path arrival
    meta: dict


class SceneRenderer:
    """Samples scenes and renders sources into them. Construct once per worker."""

    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.sr = config.sr_internal
        self.environments = config.environment_table()
        self.devices = config.device_table()
        self.contents = config.content_table()
        self.codecs = set(available_codecs())
        self.noise_banks = {
            family: AudioBank.from_list(path, self.sr, "noise")
            for family, path in config.noise.lists.items()
            if path
        }
        self.beds = (
            AudioBank.from_list(config.content.bed_list, self.sr, "bed")
            if config.content.bed_list
            else None
        )
        self.room_banks = {
            env: IRBank.from_list(path, self.sr)
            for env, path in config.room.measured.items()
            if path
        }
        self.mic_irs = (
            IRBank.from_list(config.capture.mic_ir_list, self.sr, 200.0)
            if config.capture.mic_ir_list
            else None
        )
        self.measured_responses = {
            kind: [
                load_measured_response(p, self.sr)
                for p in read_path_list(
                    path, kind="device response", suffixes=(".wav", ".flac", ".npz")
                )
            ]
            for kind, path in config.devices.measured.items()
            if path
        }

    # -- scene -----------------------------------------------------------------

    def sample_scene(self, seed: int, key: str, environment: str | None = None) -> Scene:
        rng = stream_rng(seed, "room")
        env = environment or choose(rng, self.config.environments)
        preset = self.environments[env]
        room = build_room(preset, rng)
        measured = None
        if env in self.room_banks:
            bank = self.room_banks[env]
            groups = bank.rooms()
            label = sorted(groups)[int(rng.integers(len(groups)))]
            measured = {"bank": bank, "indices": groups[label], "room": label}
        mic, mic_meta = place_mic(preset, room.dims, stream_rng(seed, "mic"))
        nrng = stream_rng(seed, "noise")
        condition = choose(nrng, {k: v["weight"] for k, v in preset["noise"].items()})
        noise_dba = sample_range(nrng, preset["noise"][condition]["level_dba"])
        cap = sample_capture(
            self.config.capture, self.sr, stream_rng(seed, "capture"), self.mic_irs, self.codecs
        )
        return Scene(
            key, seed, env, preset, room, mic, mic_meta, condition, float(noise_dba), cap, measured
        )

    def impulse_response(
        self, scene: Scene, pose: Pose, directivity: Directivity, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        distance = float(np.linalg.norm(pose.position - scene.mic))
        if scene.measured is None:
            return room_impulse_response(
                scene.room,
                pose.position,
                scene.mic,
                self.sr,
                rng,
                directivity,
                early_ms=self.config.room.early_ms,
                jitter_m=self.config.room.jitter_m,
            )
        # A measured room: a different measured position per source where possible.
        bank: IRBank = scene.measured["bank"]
        indices = scene.measured["indices"]
        free = [i for i in indices if i not in scene.used_irs] or indices
        index = free[int(rng.integers(len(free)))]
        scene.used_irs.append(index)
        delay = round(distance / SPEED_OF_SOUND * self.sr)
        h = np.concatenate(
            [np.zeros(delay), scale_for_distance(bank.get(bank.paths[index]), self.sr, distance)]
        )
        return h, {
            "measured_ir": bank.labels[index],
            "distance_m": round(distance, 3),
            "delay_s": round(distance / SPEED_OF_SOUND, 5),
        }

    # -- sources ---------------------------------------------------------------

    def render_talker(self, scene: Scene, dry: np.ndarray, index: int = 0) -> SourceRender:
        rng = stream_rng(scene.seed, "talker", index)
        placement = place_talker(scene.preset, scene.room.dims, scene.mic, rng)
        talker = sample_talker(placement, self.config.talker, scene.noise_dba, rng)
        emitted = talker_emission(dry, self.sr, talker)
        h, path = self.impulse_response(scene, placement.poses[0], talker.directivity, rng)
        return SourceRender(
            "live",
            signal.fftconvolve(emitted, h),
            path["delay_s"],
            {"role": "live", **talker.describe(), "path": path},
        )

    def render_device(
        self,
        scene: Scene,
        program: np.ndarray,
        index: int = 0,
        kind: str | None = None,
        raw_programme: bool = False,
    ) -> SourceRender:
        """Play ``program`` through a device of the scene.

        ``raw_programme`` skips the broadcast / streaming / call chain and plays the
        signal as it is - a recording replayed straight from a phone. The device and its
        placement are drawn exactly as they would be otherwise, so the same ``index``
        gives the same loudspeaker in the same spot either way.
        """
        rng = stream_rng(scene.seed, "device", index)
        kind = kind or choose(rng, scene.preset["devices"])
        preset = self.devices[kind]
        placement = place_device(kind, scene.preset, scene.room.dims, scene.mic, rng)
        content_name = choose(rng, preset["content"])
        use_measured = (
            kind in self.measured_responses and rng.random() < self.config.devices.measured_prob
        )
        device = sample_device(
            kind,
            preset,
            placement,
            self.sr,
            rng,
            scene.noise_dba,
            self.config.devices.volume_slope,
            content_name,
            self.measured_responses[kind] if use_measured else None,
        )
        if raw_programme:
            programme, content_meta = program, {"content": "raw"}
        else:
            crng = stream_rng(scene.seed, "content", index)
            chain = sample_content(
                content_name, self.contents[content_name], crng, self.codecs, bool(self.beds)
            )
            programme, content_meta = apply_content(program, self.sr, chain, crng, self.beds)
        drivers, dsp_meta = device_emission(programme, self.sr, device)

        total = np.zeros(0)
        paths = []
        for pose, directivity, driven in zip(
            placement.poses, device.directivities, drivers, strict=True
        ):
            h, path = self.impulse_response(scene, pose, directivity, rng)
            y = signal.fftconvolve(driven, h)
            if len(y) > len(total):
                total = np.pad(total, (0, len(y) - len(total)))
            total[: len(y)] += y
            paths.append(path)
        meta = {
            "role": "device",
            **device.describe(),
            "content": content_meta,
            "dsp_result": dsp_meta,
            "paths": paths,
        }
        return SourceRender("device", total, min(p["delay_s"] for p in paths), meta)

    # -- shared parts ----------------------------------------------------------

    def ambient(self, scene: Scene, n: int) -> tuple[np.ndarray, str]:
        if not self.config.noise.enabled:
            return np.zeros(n), "none"
        rng = stream_rng(scene.seed, "noise", 1)
        family = scene.preset["family"]
        bank = self.noise_banks.get(family)
        if bank:
            x, label = bank.sample(n, rng)
        else:
            x = synth_noise(family, scene.noise_condition, n, self.sr, rng)
            label = f"synthetic:{family}:{scene.noise_condition}"
        return scale_to_dba(x, self.sr, scene.noise_dba), label

    def capture(self, scene: Scene, p_mic: np.ndarray) -> tuple[np.ndarray, dict]:
        return capture(
            p_mic,
            self.sr,
            self.config.output.sr,
            scene.capture,
            stream_rng(scene.seed, "capture_noise"),
        )


__all__ = ["Scene", "SceneRenderer", "SourceRender", "derive_seed", "spl_dba", "stream_rng"]
