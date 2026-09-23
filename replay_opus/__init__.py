"""Render TTS speech as live talkers and as loudspeaker playback in shared scenes.

Training data for the rejection side of a full-duplex spoken dialogue system: the
same TTS utterances rendered once as a person talking in a home or a car and once
as a TV, phone, laptop, smart speaker or car audio system playing them back - in the
same room, with the same noise and the same microphone, so that only the emission
path separates the classes.

    from replay_opus import Config, UtteranceRenderer, render_mic

    renderer = UtteranceRenderer(Config())
    live, device = renderer.render(audio, sr, key="utt_0001", variant=0)
    mic = render_mic(speech="user.wav", playback="tv.wav")
"""

from .acoustics import Directivity, Room, absorption_for_rt60, room_impulse_response
from .compose import ComposedScene, SceneComposer, frame_labels
from .config import Config
from .mix import MicRender, render_mic
from .render import RenderedClip, UtteranceRenderer
from .scene import Scene, SceneRenderer

__version__ = "0.2.0"

__all__ = [
    "ComposedScene",
    "Config",
    "Directivity",
    "MicRender",
    "RenderedClip",
    "Room",
    "Scene",
    "SceneComposer",
    "SceneRenderer",
    "UtteranceRenderer",
    "absorption_for_rt60",
    "frame_labels",
    "render_mic",
    "room_impulse_response",
]
