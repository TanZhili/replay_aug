"""Render TTS or recorded speech as live talkers and as loudspeaker playback in shared scenes.

Training data for the rejection side of a full-duplex spoken dialogue system: the
same utterance rendered once as a person talking in a home or a car and once as a
TV, phone, laptop, smart speaker or car audio system playing it back - in the same
room, with the same noise and the same microphone, so that only the way the voice
got to the microphone separates the classes. Playback covers a produced programme
played once, a recording made in another room and played here, and a recording made
in this room and played back; all of it is labelled ``device``.

    from replay_opus import Config, UtteranceRenderer, render_mic

    renderer = UtteranceRenderer(Config())
    live, device = renderer.render(audio, sr, key="utt_0001", variant=0)
    [device] = renderer.render(audio, sr, key="rec_0001", origin="recorded")
    mic = render_mic(speech="user.wav", playback="tv.wav")
"""

from .acoustics import Directivity, Room, absorption_for_rt60, room_impulse_response
from .compose import ComposedScene, SceneComposer, frame_labels
from .config import FORMS, ORIGINS, Config, parse_forms
from .mix import MicRender, render_mic
from .render import RenderedClip, UtteranceRenderer
from .scene import Scene, SceneRenderer

__version__ = "0.3.0"

__all__ = [
    "FORMS",
    "ORIGINS",
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
    "parse_forms",
    "render_mic",
    "room_impulse_response",
]
