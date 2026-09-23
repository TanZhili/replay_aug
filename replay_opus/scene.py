"""Scenes: one room, one microphone, one noise, any number of sources.

A scene fixes everything that the classes must share - room, microphone position,
background noise, capture chain - and sources are rendered into it independently:
a live talker as a human mouth at a human level, a playback device as a programme
chain driving loudspeakers somewhere else in the same room. Each part of a scene
draws from its own seeded random stream, so the noise is the same realisation
whichever source is rendered, and a scene can be rebuilt from its seed alone.

A device clip reaches its loudspeaker in one of three forms, all labelled ``device``:

``single``      a produced programme - broadcast, radio, a podcast - through its
                delivery chain and the device, into this room;
``rerecorded``  a person recorded on a phone in another room - a short video, a voice
                message, a call - then delivered and played here. That room, that
                phone and that background are rendered as a scene of their own. An
                input that is already a room recording (origin ``recorded``) is such
                a recording and skips this stage;
``replay``      a person recorded in this very room by a phone near them, and the
                recording played back here.

The loudspeaker does not depend on the form: the device, its placement and its room
paths are drawn before any form stage, and the stages use random streams of their
own, so the three forms of one utterance play from the same device at the same spot.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .acoustics import SPEED_OF_SOUND, Directivity, Room, early_part_s, room_impulse_response
from .capture import Capture, capture, sample_capture
from .codecs import CodecSpec, available_codecs, codec_roundtrip, pick_codec
from .config import FORMS, Config, capture_config, parse_forms
from .content import apply_content, sample_content
from .dsp import P_REF, choose, fit_length, sample_range, speech_level_db
from .environment import Pose, build_room, place_device, place_mic, place_recorder, place_talker
from .irbank import IRBank, graft_tail, scale_for_distance
from .metrics import spectral_std
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
    "form": 11,
    "form_content": 12,
    "remote": 13,
    "recorder": 14,
    "recorder_capture": 15,
    "recorder_noise": 16,
    "storage": 17,
}


def derive_seed(base_seed: int, *parts) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in (base_seed, *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def stream_rng(seed: int, name: str, index: int = 0) -> np.random.Generator:
    return np.random.default_rng([int(seed), _STREAMS[name], int(index)])


def spl_dba(x_pa: np.ndarray, sr: int) -> float:
    """Active A-weighted speech level of a pressure signal, in dB SPL."""
    return speech_level_db(x_pa, sr) - 20.0 * np.log10(P_REF)


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
    delay_s: float  # earliest direct-path arrival of the voice, first room included
    meta: dict
    ir: np.ndarray | None = None  # a talker's room path, kept for the stages that reuse it


def _sum(parts: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(max(len(p) for p in parts))
    for p in parts:
        out[: len(p)] += p
    return out


class SceneRenderer:
    """Samples scenes and renders sources into them. Construct once per worker."""

    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.sr = config.sr_internal
        self.tail = int(self.sr * config.render.tail_ms / 1000.0)
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
        self.mic_irs = self._mic_bank(config.capture.mic_ir_list)
        playback = config.playback
        self.remote_capture = capture_config(
            playback.rerecorded.capture, "playback.rerecorded.capture"
        )
        self.recorder_capture = capture_config(playback.replay.capture, "playback.replay.capture")
        self.remote_mic_irs = self._mic_bank(self.remote_capture.mic_ir_list)
        self.recorder_mic_irs = self._mic_bank(self.recorder_capture.mic_ir_list)
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

    def _mic_bank(self, path: str | None) -> IRBank | None:
        return IRBank.from_list(path, self.sr, 200.0) if path else None

    # -- scene -----------------------------------------------------------------

    def sample_scene(
        self,
        seed: int,
        key: str,
        environment: str | None = None,
        *,
        capture_cfg=None,
        mic_irs: IRBank | None = None,
        talker_distance_m=None,
    ) -> Scene:
        """Draw a scene. The keyword arguments describe a phone somewhere else: its own
        capture chain and microphone responses, and how far the talker holds it."""
        rng = stream_rng(seed, "room")
        env = environment or choose(rng, self.config.environments)
        preset = self.environments[env]
        if talker_distance_m is not None and preset["layout"] != "car":
            preset = {**preset, "talker_distance_m": list(talker_distance_m)}
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
            capture_cfg or self.config.capture,
            self.sr,
            stream_rng(seed, "capture"),
            self.mic_irs if capture_cfg is None else mic_irs,
            self.codecs,
        )
        return Scene(
            key, seed, env, preset, room, mic, mic_meta, condition, float(noise_dba), cap, measured
        )

    def impulse_response(
        self,
        scene: Scene,
        pose: Pose,
        directivity: Directivity,
        rng: np.random.Generator,
        receiver: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Room path from a source to the scene's microphone, or to ``receiver``."""
        mic = scene.mic if receiver is None else np.asarray(receiver, dtype=np.float64)
        distance = float(np.linalg.norm(pose.position - mic))
        if scene.measured is None:
            h, info = self._modelled_ir(scene, pose, directivity, rng, mic)
        else:
            # A measured room: a different measured position per source where possible.
            bank: IRBank = scene.measured["bank"]
            indices = scene.measured["indices"]
            free = [i for i in indices if i not in scene.used_irs] or indices
            index = free[int(rng.integers(len(free)))]
            scene.used_irs.append(index)
            measured = bank.get(bank.paths[index])
            mode = self.config.room.measured_mode
            if mode == "full":
                delay = round(distance / SPEED_OF_SOUND * self.sr)
                h = np.concatenate(
                    [np.zeros(delay), scale_for_distance(measured, self.sr, distance)]
                )
                info = {
                    "distance_m": round(distance, 3),
                    "delay_s": round(distance / SPEED_OF_SOUND, 5),
                }
            else:
                h, info = self._modelled_ir(scene, pose, directivity, rng, mic)
                h = graft_tail(
                    h, measured, self.sr, early_part_s(scene.room, self.config.room.early_ms)
                )
            info = {**info, "measured_ir": bank.labels[index], "measured_mode": mode}
        info["sstd_db"] = round(spectral_std(h, self.sr), 2)
        return h, info

    def _modelled_ir(self, scene, pose, directivity, rng, mic) -> tuple[np.ndarray, dict]:
        return room_impulse_response(
            scene.room,
            pose.position,
            mic,
            self.sr,
            rng,
            directivity,
            early_ms=self.config.room.early_ms,
            jitter_m=self.config.room.jitter_m,
        )

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
            h,
        )

    def draw_form(self, scene: Scene, index: int = 0, weights=None) -> str:
        """Playback form of device ``index``: from ``weights`` if given (any form spec
        that :func:`replay_opus.config.parse_forms` reads), else ``playback.forms``."""
        table = self.config.playback.forms if weights is None else parse_forms(weights)
        return choose(stream_rng(scene.seed, "form", index), table)

    def render_device(
        self,
        scene: Scene,
        program: np.ndarray | None,
        index: int = 0,
        kind: str | None = None,
        form: str = "single",
        recorded: np.ndarray | None = None,
        raw_programme: bool = False,
    ) -> SourceRender:
        """Play a voice through a device of the scene.

        ``program`` is dry speech, the sound leaving a mouth. ``recorded`` is material a
        microphone has already picked up in a room (an input of origin ``recorded``);
        it can only be played in the ``rerecorded`` form, where it stands in for the
        remote recording. ``raw_programme`` plays ``program`` as it is, with no form
        stage and no delivery chain - a recording replayed straight from a phone, which
        is how ``judge`` builds its positive control.
        """
        if form not in FORMS:
            raise ValueError(f"unknown playback form {form!r}; choose from {', '.join(FORMS)}")
        if recorded is not None and form != "rerecorded":
            raise ValueError("recorded material can only be played in the 'rerecorded' form")
        if program is None and recorded is None:
            raise ValueError("render_device needs a program or recorded material")

        device, preset, drawn_content, irs, paths = self._draw_device(scene, index, kind)
        if raw_programme:
            programme, content_meta = program, {"content": "raw"}
            playback: dict = {"form": "raw"}
            stage_delay, first_room = 0.0, None
        else:
            programme, content_meta, playback, stage_delay, first_room = self._programme(
                scene, index, form, preset, drawn_content, program, recorded
            )
        drivers, dsp_meta = device_emission(programme, self.sr, device)

        total = np.zeros(0)
        for driven, h in zip(drivers, irs, strict=True):
            y = signal.fftconvolve(driven, h)
            if len(y) > len(total):
                total = np.pad(total, (0, len(y) - len(total)))
            total[: len(y)] += y
        if not raw_programme:
            acoustic = irs[0] if first_room is None else signal.fftconvolve(first_room, irs[0])
            playback["acoustic_sstd_db"] = round(spectral_std(acoustic, self.sr), 2)
        meta = {
            "role": "device",
            **device.describe(),
            "playback": playback,
            "content": content_meta,
            "dsp_result": dsp_meta,
            "paths": paths,
        }
        return SourceRender("device", total, min(p["delay_s"] for p in paths) + stage_delay, meta)

    def _draw_device(self, scene: Scene, index: int, kind: str | None):
        """Device ``index`` of the scene: its instance, preset, own content draw, and the
        room path from each of its drivers. Nothing here depends on the playback form."""
        rng = stream_rng(scene.seed, "device", index)
        kind = kind or choose(rng, scene.preset["devices"])
        preset = self.devices[kind]
        placement = place_device(kind, scene.preset, scene.room.dims, scene.mic, rng)
        drawn_content = choose(rng, preset["content"])
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
            drawn_content,
            self.measured_responses[kind] if use_measured else None,
        )
        irs, paths = [], []
        for pose, directivity in zip(placement.poses, device.directivities, strict=True):
            h, path = self.impulse_response(scene, pose, directivity, rng)
            irs.append(h)
            paths.append(path)
        return device, preset, drawn_content, irs, paths

    def _programme(
        self,
        scene: Scene,
        index: int,
        form: str,
        preset: dict,
        drawn_content: str,
        program: np.ndarray | None,
        recorded: np.ndarray | None,
    ):
        """What the device is fed: the voice after the form's recording stage and, for
        ``single`` and ``rerecorded``, a delivery chain. Returns the programme, the chain's
        description, the playback description, the delay the recording stage adds to the
        voice, and the room path of that stage (``None`` without one)."""
        if form == "replay":
            programme, stage, first_room = self._replay_recording(scene, program, index)
            content_meta = {"content": "recording", **stage.pop("storage")}
            playback = {"form": form, "origin": "dry", **stage}
            return programme, content_meta, playback, stage["talker"]["path"]["delay_s"], first_room

        content = self._form_content(scene, index, preset, form, drawn_content)
        remote = form == "rerecorded"
        playback = {"form": form, "origin": "recorded" if recorded is not None else "dry"}
        source, stage_delay, first_room = program, 0.0, None
        if remote:
            parts = [] if recorded is None else [np.asarray(recorded, dtype=np.float64)]
            if program is not None:
                recording, stage, first_room = self._remote_recording(
                    scene, program, index, content
                )
                stage_delay = stage["talker"]["path"]["delay_s"]
                parts.append(recording)
                playback["remote"] = stage
            source = _sum(parts)
        crng = stream_rng(scene.seed, "content", index)
        chain = sample_content(
            content, self.contents[content], crng, self.codecs, bool(self.beds), remote=remote
        )
        programme, content_meta = apply_content(source, self.sr, chain, crng, self.beds)
        return programme, content_meta, playback, stage_delay, first_room

    def _form_content(self, scene: Scene, index: int, preset: dict, form: str, drawn: str) -> str:
        """The device's own content draw if the form allows it, else a redraw among the
        allowed profiles the device plays; a device with none keeps its own."""
        allowed = set(getattr(self.config.playback, form).contents)
        weights = {k: w for k, w in preset["content"].items() if k in allowed and w > 0}
        if not weights or drawn in weights:
            return drawn
        return choose(stream_rng(scene.seed, "form_content", index), weights)

    def _remote_recording(
        self, scene: Scene, program: np.ndarray, index: int, content: str
    ) -> tuple[np.ndarray, dict, np.ndarray]:
        """The voice as a phone recorded it in another room: a scene of its own."""
        cfg = self.config.playback.rerecorded
        env = choose(stream_rng(scene.seed, "remote", index), cfg.environments)
        remote = self.sample_scene(
            derive_seed(scene.seed, "remote", index),
            f"{scene.key}#remote{index}",
            env,
            capture_cfg=self.remote_capture,
            mic_irs=self.remote_mic_irs,
            talker_distance_m=cfg.distance_m.get(content, cfg.distance_m["default"]),
        )
        n = len(program) + self.tail
        talker = self.render_talker(remote, program)
        noise, noise_label = self.ambient(remote, n)
        recording, capture_meta = capture(
            fit_length(talker.signal, n) + noise,
            self.sr,
            self.sr,
            remote.capture,
            stream_rng(remote.seed, "capture_noise"),
        )
        stage = {
            **remote.describe(),
            "noise_source": noise_label,
            "capture_result": capture_meta,
            "talker": talker.meta,
        }
        return recording, stage, talker.ir

    def _replay_recording(
        self, scene: Scene, program: np.ndarray, index: int
    ) -> tuple[np.ndarray, dict, np.ndarray]:
        """The voice as a phone near the talker recorded it in this very room."""
        cfg = self.config.playback.replay
        trng = stream_rng(scene.seed, "talker", 1 + index)
        placement = place_talker(scene.preset, scene.room.dims, scene.mic, trng)
        talker = sample_talker(placement, self.config.talker, scene.noise_dba, trng)
        emitted = talker_emission(program, self.sr, talker)
        rng = stream_rng(scene.seed, "recorder", index)
        pose = placement.poses[0]
        position, recorder_meta = place_recorder(
            scene.preset, scene.room.dims, pose, cfg.distance_m, rng
        )
        h, path = self.impulse_response(scene, pose, talker.directivity, rng, receiver=position)
        n = len(program) + self.tail
        noise, noise_label = self.ambient(scene, n, index=2 + index)
        cap = sample_capture(
            self.recorder_capture,
            self.sr,
            stream_rng(scene.seed, "recorder_capture", index),
            self.recorder_mic_irs,
            self.codecs,
        )
        recording, capture_meta = capture(
            fit_length(signal.fftconvolve(emitted, h), n) + noise,
            self.sr,
            self.sr,
            cap,
            stream_rng(scene.seed, "recorder_noise", index),
        )
        storage: dict = {}
        srng = stream_rng(scene.seed, "storage", index)
        codec = pick_codec(srng, cfg.codecs, self.codecs)
        if codec is not None:
            kbps = sample_range(srng, cfg.bitrate_kbps.get(codec, 64.0))
            recording, codec_meta = codec_roundtrip(
                recording, self.sr, CodecSpec(codec, int(kbps * 1000))
            )
            storage = {
                "storage_codec": codec,
                "storage_bitrate_kbps": round(kbps, 2),
                "storage_codec_sr": codec_meta["codec_sr"],
            }
        stage = {
            "recorder": {
                "pos_m": [round(float(v), 3) for v in position],
                "distance_m": round(float(np.linalg.norm(position - pose.position)), 3),
                **recorder_meta,
                "capture": cap.describe(),
                "capture_result": capture_meta,
                "noise_source": noise_label,
            },
            "talker": {**talker.describe(), "path": path},
            "storage": storage,
        }
        return recording, stage, h

    # -- shared parts ----------------------------------------------------------

    def ambient(self, scene: Scene, n: int, index: int = 1) -> tuple[np.ndarray, str]:
        """The scene's background at the microphone. Other ``index`` values give the
        same noise conditions at another moment - what a recorder heard earlier."""
        if not self.config.noise.enabled:
            return np.zeros(n), "none"
        rng = stream_rng(scene.seed, "noise", index)
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
