"""Scene rendering: what the two classes must share, and what may differ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from replay_opus.config import Config
from replay_opus.environment import ENVIRONMENTS
from replay_opus.irbank import synth_ir
from replay_opus.render import UtteranceRenderer
from replay_opus.scene import SceneRenderer

from .conftest import SR, speech_like


def test_paired_render_shares_scene_noise_and_capture(speech):
    config = Config()
    config.capture.agc_prob = 0.0  # a causal chain, so the lead-in must match exactly
    # Programmes that stay silent before the speech: no codec floor, no call noise.
    config.content.weights = {"broadcast": 1.0}
    config.content.profiles = {"broadcast": {"codecs": {"none": 1.0}, "source_room_prob": 0.0}}
    renderer = UtteranceRenderer(config)
    for variant in range(4):
        live, device = renderer.render(speech, SR, "utt", variant)
        assert live.label == "live" and device.label == "device"
        assert len(live.audio) == len(device.audio)
        assert live.meta["scene"] == device.meta["scene"]
        lead = int(0.2 * live.sr)  # speech starts at 0.3 s
        assert np.allclose(live.audio[:lead], device.audio[:lead], atol=1e-6)
        assert not np.allclose(live.audio, device.audio)


def test_render_is_deterministic_and_variants_differ(speech):
    renderer = UtteranceRenderer(Config())
    first = renderer.render(speech, SR, "utt", 0)
    again = renderer.render(speech, SR, "utt", 0)
    other = renderer.render(speech, SR, "utt", 1)
    for a, b in zip(first, again, strict=True):
        assert np.array_equal(a.audio, b.audio)
    assert first[0].meta["scene"] != other[0].meta["scene"]


@pytest.mark.parametrize("mode", ["live", "device", "mixed"])
def test_single_class_modes(mode, speech):
    config = Config()
    config.render.mode = mode
    clips = UtteranceRenderer(config).render(speech, SR, "utt", 0)
    assert len(clips) == 1
    if mode != "mixed":
        assert clips[0].label == mode


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_every_environment_renders_with_sensible_levels(env, speech):
    config = Config()
    config.environments = {env: 1.0}
    live, device = UtteranceRenderer(config).render(speech, SR, "utt", 0)
    for clip in (live, device):
        assert clip.meta["scene"]["environment"] == env
        assert 35.0 < clip.meta["at_mic"]["speech_dba"] < 100.0
        assert np.max(np.abs(clip.audio)) <= 1.0
        assert clip.sr == config.output.sr
    assert live.meta["source"]["role"] == "live"
    assert device.meta["source"]["role"] == "device"


def test_measured_rooms_come_from_one_room_per_scene(tmp_path, rng, speech):
    lines = []
    for room in ("room_a", "room_b"):
        for pos in range(3):
            path = tmp_path / "irs" / room / f"pos{pos}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, synth_ir(SR, 0.4, 6.0, rng), SR)
            lines.append(f"irs/{room}/pos{pos}.wav")
    listing = tmp_path / "irs.txt"
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = Config()
    config.environments = {"living_room": 1.0}
    config.room.measured = {"living_room": str(listing)}
    for variant in range(3):
        live, device = UtteranceRenderer(config).render(speech, SR, "utt", variant)
        used = [live.meta["source"]["path"]["measured_ir"]] + [
            p["measured_ir"] for p in device.meta["source"]["paths"]
        ]
        assert len({u.split("/")[0] for u in used}) == 1  # one room per scene
        assert live.meta["scene"]["measured_room"] == used[0].split("/")[0]


def test_noise_list_bed_list_and_mic_irs_are_used(tmp_path, rng, speech):
    noise = tmp_path / "noise" / "hum.wav"
    noise.parent.mkdir()
    sf.write(noise, 0.1 * rng.standard_normal(2 * SR), SR)
    (tmp_path / "noise.txt").write_text("noise/hum.wav\n", encoding="utf-8")
    bed = tmp_path / "beds" / "music.wav"
    bed.parent.mkdir()
    sf.write(bed, 0.1 * np.sin(2 * np.pi * 220 * np.arange(3 * SR) / SR), SR)
    (tmp_path / "beds.txt").write_text("beds/music.wav\n", encoding="utf-8")
    mic = tmp_path / "mics" / "phone.wav"
    mic.parent.mkdir()
    sf.write(mic, np.eye(1, 64, 3).ravel(), SR)
    (tmp_path / "mics.txt").write_text("mics/phone.wav\n", encoding="utf-8")

    config = Config()
    config.environments = {"living_room": 1.0}
    config.noise.lists = {"home": str(tmp_path / "noise.txt")}
    config.content.bed_list = str(tmp_path / "beds.txt")
    config.content.profiles = {
        name: {"bed_prob": 1.0} for name in ("broadcast", "streaming", "radio")
    }
    config.devices.weights = {"tv": 1.0}
    config.capture.mic_ir_list = str(tmp_path / "mics.txt")
    live, device = UtteranceRenderer(config).render(speech, SR, "utt", 0)
    assert live.meta["scene"]["noise_source"] == "hum.wav"
    assert live.meta["scene"]["capture"]["mic_ir"] == "phone.wav"
    assert device.meta["source"]["content"].get("bed") == "music.wav"


def test_measured_device_responses_replace_the_model(tmp_path, rng, speech):
    response = tmp_path / "resp" / "my_tv.wav"
    response.parent.mkdir()
    h = np.zeros(2048)
    h[10] = 1.0
    h[11:200] = 0.05 * rng.standard_normal(189)
    sf.write(response, h, SR)
    (tmp_path / "resp.txt").write_text("resp/my_tv.wav\n", encoding="utf-8")
    config = Config()
    config.environments = {"living_room": 1.0}
    config.devices.weights = {"tv": 1.0}
    config.devices.measured = {"tv": str(tmp_path / "resp.txt")}
    config.devices.measured_prob = 1.0
    _, device = UtteranceRenderer(config).render(speech, SR, "utt", 0)
    assert device.meta["source"]["measured_response"] is True


def test_config_rejects_bad_settings():
    with pytest.raises(ValueError, match="unknown config key"):
        Config.from_dict({"room": {"nope": 1}})
    with pytest.raises(ValueError, match="unknown environment"):
        Config.from_dict({"environments": {"garage": 1.0}}).validate()
    with pytest.raises(ValueError, match=r"render\.mode"):
        Config.from_dict({"render": {"mode": "both"}}).validate()
    with pytest.raises(ValueError, match="no preset or placement"):
        Config.from_dict(
            {"environment_presets": {"living_room": {"devices": {"car_audio": 1.0}}}}
        ).validate()


def test_preset_overrides_merge_over_the_builtins():
    config = Config.from_dict(
        {
            "devices": {"presets": {"tv": {"level_dba": [70.0, 71.0]}}},
            "environment_presets": {"garage": {**ENVIRONMENTS["kitchen"], "rt60_s": [1.0, 1.5]}},
            "environments": {"garage": 1.0},
        }
    )
    config.validate()
    assert config.device_table()["tv"]["level_dba"] == [70.0, 71.0]
    assert config.device_table()["tv"]["hp_hz"] == [90.0, 220.0]
    scene = SceneRenderer(config).sample_scene(1, "k")
    assert scene.environment == "garage"
    assert scene.room.rt60()[3] > 0.8


def test_default_yaml_is_the_builtin_default():
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    assert Config.load(path).to_dict() == Config().to_dict()


def test_speech_like_fixture_has_silence_around_it():
    x = speech_like()
    assert np.all(x[: int(0.3 * SR)] == 0.0)
