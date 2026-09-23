"""Full-duplex training scenes: device programmes and live users on one timeline.

A full-duplex model does not hear isolated clips. It hears a stream in which a TV
talks on and off, a user speaks now and then - sometimes over the TV - and the room
is never silent. Each composed scene is such a stream with time-aligned labels:
which stretches are a live user, which are a device, and which device.

Timeline rules, all set in the ``scene`` config section:

* scene types ``both`` / ``device_only`` / ``live_only`` / ``empty``, by weight;
* a device programme is either continuous - a TV that was already on: utterances,
  possibly from several voices, with short gaps, cut to the scene - or a burst, like
  a phone video that starts and stops inside the scene;
* one to three user utterances at random times, free to overlap the programme.

Labels come in two lists, both in microphone time (placement plus direct-path delay)
and cut to the scene: ``utterances``, one per placed utterance with its text, speaker
and device, and ``segments``, the voice activity inside them, each pointing back to
its utterance.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from .capture import to_digital
from .config import Config
from .dsp import (
    choose,
    fade_edges,
    fit_length,
    resample_to,
    sample_int,
    sample_range,
    speech_segments,
    to_mono,
)
from .pathlist import Utterance
from .scene import SceneRenderer, derive_seed, spl_dba, stream_rng

MIN_SEGMENT_S = 0.05


@dataclass
class ComposedScene:
    audio: np.ndarray
    sr: int
    utterances: list[dict]  # one per placed utterance: role, source, times, text, speaker
    segments: list[dict]  # voice activity inside them, each pointing to its utterance
    meta: dict
    stems: dict[str, np.ndarray] = field(default_factory=dict)


class SceneComposer:
    def __init__(self, config: Config, utterances: list[Utterance]):
        if not utterances:
            raise ValueError("scene composition needs at least one utterance")
        self.config = config
        self.scenes = SceneRenderer(config)
        self.sr = config.sr_internal
        self.utterances = utterances
        self._cache: OrderedDict[int, tuple[np.ndarray, list[tuple[float, float]]]] = OrderedDict()
        by_speaker: dict[str, list[int]] = defaultdict(list)
        for index, utt in enumerate(utterances):
            if utt.speaker is not None:
                by_speaker[utt.speaker].append(index)
        # Voice-aware choices only when every utterance says who is speaking.
        self.by_speaker = (
            dict(by_speaker) if len(by_speaker) > 1 and all(u.speaker for u in utterances) else {}
        )

    # -- material ----------------------------------------------------------------

    def _dry(self, index: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
        if index not in self._cache:
            if len(self._cache) >= 256:
                self._cache.popitem(last=False)
            audio, sr = sf.read(str(self.utterances[index].path), always_2d=False)
            dry = fade_edges(resample_to(to_mono(audio), sr, self.sr), self.sr, 5.0)
            self._cache[index] = (dry, speech_segments(dry, self.sr))
        self._cache.move_to_end(index)
        return self._cache[index]

    def _pool(self, rng: np.random.Generator, n_voices: int, exclude: set[str]) -> list[int]:
        if not self.by_speaker:
            return list(range(len(self.utterances)))
        voices = [v for v in self.by_speaker if v not in exclude] or list(self.by_speaker)
        chosen = rng.choice(len(voices), size=min(n_voices, len(voices)), replace=False)
        return [i for c in chosen for i in self.by_speaker[voices[int(c)]]]

    def _place(self, track: np.ndarray, index: int, start: int, placed: list[dict]) -> None:
        """Add an utterance to a track and note where its speech lies (in track time)."""
        dry, speech = self._dry(index)
        lo, hi = max(start, 0), min(start + len(dry), len(track))
        if hi <= lo:
            return
        track[lo:hi] += dry[lo - start : hi - start]
        utt = self.utterances[index]
        offset = start / self.sr
        parts = [(offset + s0, offset + s1) for s0, s1 in speech]
        placed.append({"utt": utt.key, "text": utt.text, "speaker": utt.speaker, "parts": parts})

    def _programme(self, n: int, rng: np.random.Generator, continuous: bool, exclude: set[str]):
        cfg = self.config.scene
        pool = self._pool(rng, sample_int(rng, cfg.device_voices), exclude)
        track = np.zeros(n)
        placed: list[dict] = []
        if continuous:
            cursor = -int(rng.uniform(0.0, 4.0) * self.sr)  # the TV was already on
            end = n
        else:
            cursor = int(rng.uniform(0.0, 0.6) * n)
            end = min(n, cursor + int(rng.uniform(2.0, 8.0) * self.sr))
        while cursor < end:
            index = pool[int(rng.integers(len(pool)))]
            self._place(track, index, cursor, placed)
            cursor += len(self._dry(index)[0]) + int(sample_range(rng, cfg.device_gap_s) * self.sr)
        voices = sorted({p["speaker"] for p in placed if p["speaker"]})
        return track, placed, voices

    def _live_track(self, n: int, rng: np.random.Generator, exclude: set[str]):
        cfg = self.config.scene
        pool = self._pool(rng, 1, exclude)
        picks = [
            pool[int(rng.integers(len(pool)))] for _ in range(sample_int(rng, cfg.live_utterances))
        ]
        gaps_min = float(cfg.live_min_gap_s)
        while (
            picks
            and sum(len(self._dry(i)[0]) for i in picks) + (len(picks) + 1) * gaps_min * self.sr > n
        ):
            picks.pop()
        track = np.zeros(n)
        placed: list[dict] = []
        if not picks:
            return track, placed
        lengths = [len(self._dry(i)[0]) for i in picks]
        free = n - sum(lengths) - int((len(picks) + 1) * gaps_min * self.sr)
        cuts = np.sort(rng.uniform(0, free, len(picks)))
        cursor = int(gaps_min * self.sr)
        previous = 0.0
        for index, length, cut in zip(picks, lengths, cuts, strict=True):
            cursor += int(cut - previous)
            previous = cut
            self._place(track, index, cursor, placed)
            cursor += length + int(gaps_min * self.sr)
        return track, placed

    # -- scene -----------------------------------------------------------------

    def compose(self, index: int) -> ComposedScene:
        cfg = self.config.scene
        seed = derive_seed(self.config.seed, "scene", index)
        rng = stream_rng(seed, "compose")
        scene = self.scenes.sample_scene(seed, f"scene_{index:06d}")
        duration = sample_range(rng, cfg.duration_s)
        n = int(duration * self.sr)
        kind = choose(rng, cfg.types)

        mix = np.zeros(n)
        stems: dict[str, np.ndarray] = {}
        placed: list[dict] = []
        sources: dict[str, dict] = {}
        voices_used: set[str] = set()

        if kind in ("both", "device_only"):
            n_devices = 2 if rng.random() < cfg.second_device_prob else 1
            for d in range(n_devices):
                continuous = rng.random() < cfg.device_continuous_prob
                track, programme, voices = self._programme(n, rng, continuous, set())
                voices_used.update(voices)
                source = self.scenes.render_device(scene, track, index=d)
                signal = fit_length(source.signal, n)
                name = f"device{d}"
                mix += signal
                stems[name] = signal
                extra = {
                    "role": "device",
                    "source": name,
                    "device": source.meta["device"],
                    "content": source.meta["content"]["content"],
                }
                placed += [{**p, **extra, "delay": source.delay_s} for p in programme]
                sources[name] = {
                    **source.meta,
                    "continuous": continuous,
                    "speech_dba_at_mic": round(spl_dba(signal, self.sr), 2) if programme else None,
                }

        if kind in ("both", "live_only"):
            track, spoken = self._live_track(n, rng, voices_used)
            if spoken:
                source = self.scenes.render_talker(scene, track)
                signal = fit_length(source.signal, n)
                mix += signal
                stems["live0"] = signal
                placed += [
                    {**p, "role": "live", "source": "live0", "delay": source.delay_s}
                    for p in spoken
                ]
                sources["live0"] = {
                    **source.meta,
                    "speech_dba_at_mic": round(spl_dba(signal, self.sr), 2),
                }

        noise, noise_label = self.scenes.ambient(scene, n)
        audio, capture_meta = self.scenes.capture(scene, mix + noise)
        out_sr = self.config.output.sr
        utterances, segments = _label(placed, duration)

        stem_audio: dict[str, np.ndarray] = {}
        if cfg.save_stems:
            stems["noise"] = noise
            for name, stem in stems.items():
                stem_audio[name] = resample_to(
                    to_digital(stem, self.sr, scene.capture), self.sr, out_sr
                )

        present = sorted({s["role"] for s in segments})
        meta = {
            "scene_type": kind,
            "roles_present": present,
            "duration_s": round(len(audio) / out_sr, 3),
            "sr": out_sr,
            "utterances": utterances,
            "segments": segments,
            "sources": sources,
            "scene": {**scene.describe(), "noise_source": noise_label},
            "capture_result": capture_meta,
        }
        return ComposedScene(audio, out_sr, utterances, segments, meta, stem_audio)


def _label(placed: list[dict], duration: float) -> tuple[list[dict], list[dict]]:
    """Utterance list (one entry per placed utterance, with its text) and the
    voice-activity segments inside them, both in microphone time and cut to the scene.

    A TTS utterance with long pauses splits into several segments; they all point to
    their utterance by index instead of repeating its text.
    """
    utterances = []
    for p in placed:
        parts = [
            (a + p["delay"], min(b + p["delay"], duration))
            for a, b in p["parts"]
            if min(b + p["delay"], duration) - max(a + p["delay"], 0.0) >= MIN_SEGMENT_S
        ]
        parts = [(max(a, 0.0), b) for a, b in parts]
        if parts:
            info = {k: v for k, v in p.items() if k not in {"parts", "delay"}}
            utterances.append({"start": parts[0][0], "end": parts[-1][1], **info, "parts": parts})
    utterances.sort(key=lambda u: (u["start"], u["source"]))
    segments = []
    for index, u in enumerate(utterances):
        for a, b in u.pop("parts"):
            segments.append(
                {
                    "start": round(a, 3),
                    "end": round(b, 3),
                    "role": u["role"],
                    "source": u["source"],
                    "utterance": index,
                }
            )
        u["start"], u["end"] = round(u["start"], 3), round(u["end"], 3)
    segments.sort(key=lambda s: (s["start"], s["source"]))
    return utterances, segments


def frame_labels(segments: list[dict], duration_s: float, frame_s: float = 0.01) -> np.ndarray:
    """Per-frame targets from a scene's segments: bit 0 = live user, bit 1 = device.

    So 0 is neither, 1 live only, 2 device only, 3 both at once.
    """
    n = int(np.ceil(duration_s / frame_s))
    labels = np.zeros(n, dtype=np.int8)
    for s in segments:
        bit = 1 if s["role"] == "live" else 2
        a = int(np.floor(s["start"] / frame_s + 1e-9))
        b = int(np.ceil(s["end"] / frame_s - 1e-9))
        labels[a:b] |= bit
    return labels


__all__ = ["ComposedScene", "SceneComposer", "frame_labels"]
