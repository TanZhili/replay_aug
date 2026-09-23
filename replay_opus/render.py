"""Per-utterance rendering: the same words as a live talker and as device playback.

In ``paired`` mode every variant samples one scene and renders the utterance into it
twice - once from a talker's mouth, once through a playback device - with the same
noise realisation and the same capture chain. The two files differ only in how the
speech was emitted, which is exactly the contrast a rejection model has to learn,
and they contain the same words and voice, so neither content nor speaker identity
can separate them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .dsp import fade_edges, fit_length, resample_to, to_mono
from .scene import SceneRenderer, derive_seed, spl_dba, stream_rng

LABELS = ("live", "device")


@dataclass
class RenderedClip:
    label: str  # "live" or "device"
    audio: np.ndarray
    sr: int
    meta: dict


class UtteranceRenderer:
    def __init__(self, config: Config):
        self.config = config
        self.scenes = SceneRenderer(config)
        self.sr = config.sr_internal

    def labels_for(self, seed: int) -> list[str]:
        mode = self.config.render.mode
        if mode == "paired":
            return list(LABELS)
        if mode in LABELS:
            return [mode]
        rng = stream_rng(seed, "mode")
        return ["device" if rng.random() < self.config.render.device_prob else "live"]

    def render(
        self, audio: np.ndarray, sr_in: int, key: str, variant: int = 0
    ) -> list[RenderedClip]:
        seed = derive_seed(self.config.seed, key, variant)
        scene = self.scenes.sample_scene(seed, key)
        dry = fade_edges(resample_to(to_mono(audio), sr_in, self.sr), self.sr, 5.0)
        n = len(dry) + int(self.sr * self.config.render.tail_ms / 1000.0)
        noise, noise_label = self.scenes.ambient(scene, n)
        scene_meta = {**scene.describe(), "noise_source": noise_label}

        clips = []
        for label in self.labels_for(seed):
            if label == "live":
                source = self.scenes.render_talker(scene, dry)
            else:
                source = self.scenes.render_device(scene, dry)
            at_mic = fit_length(source.signal, n)
            out, capture_meta = self.scenes.capture(scene, at_mic + noise)
            speech = spl_dba(at_mic, self.sr)
            clips.append(
                RenderedClip(
                    label=label,
                    audio=out,
                    sr=self.config.output.sr,
                    meta={
                        "label": label,
                        "key": key,
                        "variant": variant,
                        "sr": self.config.output.sr,
                        "duration_s": round(len(out) / self.config.output.sr, 3),
                        "at_mic": {
                            "speech_dba": round(speech, 2),
                            "snr_db": round(speech - scene.noise_dba, 2),
                            "delay_s": round(source.delay_s, 5),
                        },
                        "scene": scene_meta,
                        "source": source.meta,
                        "capture_result": capture_meta,
                    },
                )
            )
        return clips


__all__ = ["LABELS", "RenderedClip", "UtteranceRenderer"]
