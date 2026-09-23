"""Per-input rendering: the same words as a live talker and as device playback.

In ``paired`` mode every variant samples one scene and renders the input into it
twice - once from a talker's mouth, once through a playback device - with the same
noise realisation and the same capture chain. The two files differ only in how the
speech was emitted, which is exactly the contrast a rejection model has to learn,
and they hold the same words and voice, so neither content nor speaker identity can
separate them.

The device clip takes its playback form from the input's own ``playback`` field or
from ``playback.forms``. An input of origin ``recorded`` was already picked up by a
microphone in a room, so it can only become device playback: the ``rerecorded``
form, with the recording itself in place of the remote stage. Its live side is left
out, or with ``render.recorded_live: passthrough`` written as it is - through the
scene's digital front end and at its device clip's level, so that level says nothing
about the label. That live clip comes from another room and another microphone than
its device clip; see the documentation before training on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capture import digital_front_end
from .config import ORIGINS, Config, parse_forms
from .dsp import active_level_db, fade_edges, fit_length, resample_to, to_mono
from .scene import SceneRenderer, SourceRender, derive_seed, spl_dba, stream_rng

LABELS = ("live", "device")


@dataclass
class RenderedClip:
    label: str  # "live" or "device"
    audio: np.ndarray
    sr: int
    meta: dict


def resolve_input(config: Config, origin: str | None, playback) -> tuple[str, dict | None]:
    """An input's origin and playback forms, checked against the config.

    ``origin`` and ``playback`` are the input's own fields (``None`` when it has none).
    Raises ``ValueError`` for a combination that cannot be rendered.
    """
    origin = origin or config.render.origin
    if origin not in ORIGINS:
        raise ValueError(f"origin must be one of {', '.join(ORIGINS)}, got {origin!r}")
    forms = parse_forms(playback, "playback") if playback is not None else None
    if origin == "recorded":
        if forms and any(w > 0 and name != "rerecorded" for name, w in forms.items()):
            raise ValueError(
                "an input of origin 'recorded' was already recorded in a room; it can only be "
                "played back in the 'rerecorded' form"
            )
        if config.render.mode == "live" and config.render.recorded_live == "skip":
            raise ValueError(
                "render.mode 'live' renders nothing for a recorded input unless "
                "render.recorded_live is 'passthrough'"
            )
    return origin, forms


class UtteranceRenderer:
    def __init__(self, config: Config):
        self.config = config
        self.scenes = SceneRenderer(config)
        self.sr = config.sr_internal

    def labels_for(self, seed: int, origin: str = "dry") -> list[str]:
        mode = self.config.render.mode
        if mode == "paired":
            labels = list(LABELS)
        elif mode in LABELS:
            labels = [mode]
        else:
            rng = stream_rng(seed, "mode")
            labels = ["device" if rng.random() < self.config.render.device_prob else "live"]
        if origin == "recorded" and self.config.render.recorded_live == "skip":
            return ["device"]
        return labels

    def render(
        self,
        audio: np.ndarray,
        sr_in: int,
        key: str,
        variant: int = 0,
        origin: str | None = None,
        playback=None,
    ) -> list[RenderedClip]:
        """Render one input. ``origin`` and ``playback`` are the input's own fields."""
        origin, forms = resolve_input(self.config, origin, playback)
        seed = derive_seed(self.config.seed, key, variant)
        scene = self.scenes.sample_scene(seed, key)
        x = fade_edges(resample_to(to_mono(audio), sr_in, self.sr), self.sr, 5.0)
        # A recording already ends in its own room tone; a tail of ours would be silence.
        n = len(x) + (self.scenes.tail if origin == "dry" else 0)
        noise, noise_label = self.scenes.ambient(scene, n)
        scene_meta = {**scene.describe(), "noise_source": noise_label}
        base = {"key": key, "variant": variant, "origin": origin, "sr": self.config.output.sr}

        labels = self.labels_for(seed, origin)
        device = None
        if "device" in labels or origin == "recorded":
            if origin == "recorded":
                form = "rerecorded"
                source = self.scenes.render_device(scene, None, form=form, recorded=x)
            else:
                form = self.scenes.draw_form(scene, 0, forms)
                source = self.scenes.render_device(scene, x, form=form)
            tags = {**base, "playback_form": form}
            device = self._clip(scene, source, n, noise, tags, scene_meta)
        clips = []
        for label in labels:
            if label == "device":
                clips.append(device)
            elif origin == "dry":
                source = self.scenes.render_talker(scene, x)
                clips.append(self._clip(scene, source, n, noise, base, scene_meta))
            else:
                clips.append(self._passthrough(scene, x, device, base, scene_meta))
        return clips

    def _clip(
        self, scene, source: SourceRender, n: int, noise, base: dict, scene_meta: dict
    ) -> RenderedClip:
        at_mic = fit_length(source.signal, n)
        out, capture_meta = self.scenes.capture(scene, at_mic + noise)
        speech = spl_dba(at_mic, self.sr)
        return RenderedClip(
            label=source.role,
            audio=out,
            sr=self.config.output.sr,
            meta={
                "label": source.role,
                **base,
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

    def _passthrough(
        self, scene, x: np.ndarray, device: RenderedClip, base: dict, scene_meta: dict
    ) -> RenderedClip:
        out_sr = self.config.output.sr
        y, front_meta = digital_front_end(x, self.sr, out_sr, scene.capture)
        y = fit_length(y, len(device.audio))
        gain_db = active_level_db(device.audio, out_sr, weighted=False) - active_level_db(
            y, out_sr, weighted=False
        )
        y = np.clip(y * 10.0 ** (gain_db / 20.0), -1.0, 1.0)
        return RenderedClip(
            label="live",
            audio=y,
            sr=out_sr,
            meta={
                "label": "live",
                **base,
                "duration_s": round(len(y) / out_sr, 3),
                "at_mic": None,
                "scene": scene_meta,
                "source": {
                    "role": "live",
                    "passthrough": True,
                    "level_match_db": round(gain_db, 2),
                },
                "capture_result": front_meta,
            },
        )


__all__ = ["LABELS", "RenderedClip", "UtteranceRenderer", "resolve_input"]
