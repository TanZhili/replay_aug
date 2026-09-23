"""Full-duplex scenes: labels have to say where each source really is."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from replay_opus.compose import SceneComposer, frame_labels
from replay_opus.config import Config
from replay_opus.pathlist import Utterance

from .conftest import SR, speech_like


@pytest.fixture
def utterances(tmp_path) -> list[Utterance]:
    out = []
    for i in range(4):
        path = tmp_path / f"u{i}.wav"
        sf.write(path, speech_like(seconds=1.0 + 0.3 * i, seed=i, f0=120 + 30 * i), SR)
        out.append(Utterance(Path(path), f"u{i}.wav", f"text {i}", f"spk{i % 2}"))
    return out


def _config(**scene) -> Config:
    config = Config()
    config.scene.duration_s = [6.0, 8.0]
    config.scene.save_stems = True
    for key, value in scene.items():
        setattr(config.scene, key, value)
    return config


def test_segments_sit_where_the_stems_have_energy(utterances):
    config = _config(types={"both": 1.0})
    config.playback.forms = {"single": 1.0}  # recorded forms play their room tone between words
    composer = SceneComposer(config, utterances)
    checked = 0
    for index in range(4):
        scene = composer.compose(index)
        for name, stem in scene.stems.items():
            if name == "noise":
                continue
            mask = np.zeros(len(stem), bool)
            for s in scene.segments:
                if s["source"] == name:
                    mask[int(s["start"] * scene.sr) : int(s["end"] * scene.sr)] = True
            if mask.any() and (~mask).any():
                inside = np.mean(stem[mask] ** 2)
                outside = np.mean(stem[~mask] ** 2)
                assert 10 * np.log10(inside / (outside + 1e-20)) > 10.0
                checked += 1
    assert checked >= 4


def test_scene_types_decide_which_roles_appear(utterances):
    for kind, roles in (("device_only", {"device"}), ("live_only", {"live"}), ("empty", set())):
        scene = SceneComposer(_config(types={kind: 1.0}), utterances).compose(0)
        assert {s["role"] for s in scene.segments} == roles
        assert scene.meta["scene_type"] == kind
        assert len(scene.audio) == pytest.approx(scene.meta["duration_s"] * scene.sr, abs=1)


def test_utterances_carry_text_and_segments_point_to_them(utterances):
    scene = SceneComposer(_config(types={"both": 1.0}), utterances).compose(1)
    assert scene.utterances and scene.segments
    for u in scene.utterances:
        assert u["text"].startswith("text ")
        assert 0.0 <= u["start"] < u["end"] <= scene.meta["duration_s"] + 1e-6
        if u["role"] == "device":
            assert u["device"] and u["content"]
    for s in scene.segments:
        owner = scene.utterances[s["utterance"]]
        assert owner["role"] == s["role"] and owner["source"] == s["source"]
        assert owner["start"] <= s["start"] < s["end"] <= owner["end"]
        assert "text" not in s


def test_live_voice_differs_from_programme_voices(utterances):
    composer = SceneComposer(_config(types={"both": 1.0}, device_voices=[1, 1]), utterances)
    for index in range(4):
        utts = composer.compose(index).utterances
        live = {u["speaker"] for u in utts if u["role"] == "live"}
        device = {u["speaker"] for u in utts if u["role"] == "device"}
        if live and device:
            assert not live & device


def test_frame_labels_encode_overlap():
    segments = [
        {"start": 0.0, "end": 0.5, "role": "device"},
        {"start": 0.3, "end": 0.8, "role": "live"},
    ]
    labels = frame_labels(segments, 1.0, 0.1)
    assert labels.tolist() == [2, 2, 2, 3, 3, 1, 1, 1, 0, 0]


def test_composition_is_deterministic(utterances):
    config = _config()
    a = SceneComposer(config, utterances).compose(3)
    b = SceneComposer(config, utterances).compose(3)
    assert np.array_equal(a.audio, b.audio)
    assert a.segments == b.segments and a.utterances == b.utterances
