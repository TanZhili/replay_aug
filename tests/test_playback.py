"""Playback forms and input origins: one label, three ways a voice reaches a loudspeaker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from replay_opus.acoustics import Directivity, Room, absorption_for_rt60, room_impulse_response
from replay_opus.audit import run_audit
from replay_opus.cli import main
from replay_opus.compose import SceneComposer
from replay_opus.config import FORMS, Config, parse_forms
from replay_opus.denoise import suppress_noise
from replay_opus.dsp import active_level_db, speech_level_db
from replay_opus.irbank import graft_tail, synth_ir
from replay_opus.metrics import estimate_drr, estimate_t60, spectral_std
from replay_opus.mix import main as mix_main
from replay_opus.mix import render_mic
from replay_opus.pathlist import Utterance, read_utterances
from replay_opus.render import UtteranceRenderer

from .conftest import SR, speech_like


def _only(form: str, **config_fields) -> UtteranceRenderer:
    config = Config()
    config.playback.forms = {form: 1.0}
    for name, value in config_fields.items():
        setattr(config, name, value)
    return UtteranceRenderer(config)


# -- forms ---------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_every_form_is_labelled_device_and_tagged(form, speech):
    live, device = _only(form).render(speech, SR, "utt", 0)
    assert (live.label, device.label) == ("live", "device")
    assert device.meta["playback_form"] == form
    assert device.meta["source"]["playback"]["form"] == form
    assert "playback_form" not in live.meta
    assert len(live.audio) == len(device.audio)
    assert np.max(np.abs(device.audio)) <= 1.0
    assert np.isfinite(device.meta["source"]["playback"]["acoustic_sstd_db"])


def test_the_form_never_changes_the_loudspeaker(speech):
    for variant in range(3):
        looks = set()
        for form in FORMS:
            _, device = _only(form).render(speech, SR, "utt", variant)
            source = device.meta["source"]
            looks.add(repr((source["device"], source["drivers"], source["paths"])))
        assert len(looks) == 1


def test_single_plays_produced_programmes(speech):
    renderer = _only("single")
    allowed = set(renderer.config.playback.single.contents)
    for variant in range(6):
        _, device = renderer.render(speech, SR, "utt", variant)
        assert device.meta["source"]["content"]["content"] in allowed
        assert "remote" not in device.meta["source"]["playback"]


def test_rerecorded_passes_through_another_room_and_phone_first(speech):
    renderer = _only("rerecorded", environments={"living_room": 1.0})
    config = renderer.config.playback.rerecorded
    for variant in range(4):
        _, device = renderer.render(speech, SR, "utt", variant)
        playback = device.meta["source"]["playback"]
        remote = playback["remote"]
        assert remote["environment"] in config.environments
        assert remote["room"] != device.meta["scene"]["room"]
        assert {"capture", "talker", "noise_dba"} <= set(remote)
        content = device.meta["source"]["content"]
        assert content["content"] in set(config.contents)
        assert "source_room" not in content and "far_end_snr_db" not in content
        assert playback["origin"] == "dry"


def test_replay_is_recorded_in_the_same_room_by_a_phone_near_the_talker(speech):
    renderer = _only("replay")
    _, hi = renderer.config.playback.replay.distance_m
    for variant in range(4):
        live, device = renderer.render(speech, SR, "utt", variant)
        playback = device.meta["source"]["playback"]
        recorder = playback["recorder"]
        dims = np.array(device.meta["scene"]["room"]["dims_m"])
        position = np.array(recorder["pos_m"])
        assert np.all(position >= 0.0) and np.all(position <= dims)
        assert recorder["distance_m"] <= hi + 1e-6
        assert device.meta["source"]["content"]["content"] == "recording"
        assert playback["talker"]["pose"] != live.meta["source"]["pose"]  # another moment


def test_forms_come_from_the_config_weights_and_are_deterministic(speech):
    renderer = UtteranceRenderer(Config())
    forms = [renderer.render(speech, SR, "utt", v)[1].meta["playback_form"] for v in range(24)]
    assert set(forms) == set(FORMS)
    again = [renderer.render(speech, SR, "utt", v)[1].meta["playback_form"] for v in range(24)]
    assert forms == again


def test_an_input_can_choose_its_own_forms(speech):
    renderer = UtteranceRenderer(Config())
    for variant in range(4):
        _, device = renderer.render(speech, SR, "utt", variant, playback="replay")
        assert device.meta["playback_form"] == "replay"
    both = {
        renderer.render(speech, SR, "utt", v, playback=["single", "rerecorded"])[1].meta[
            "playback_form"
        ]
        for v in range(12)
    }
    assert both == {"single", "rerecorded"}


# -- recorded inputs ------------------------------------------------------------


def test_recorded_inputs_become_rerecorded_playback_only(speech):
    renderer = UtteranceRenderer(Config())
    clips = renderer.render(speech, SR, "rec", 0, origin="recorded")
    assert [c.label for c in clips] == ["device"]
    device = clips[0]
    playback = device.meta["source"]["playback"]
    assert device.meta["playback_form"] == "rerecorded" and device.meta["origin"] == "recorded"
    assert playback["origin"] == "recorded" and "remote" not in playback
    assert device.meta["duration_s"] == pytest.approx(len(speech) / SR, abs=0.01)


def test_recorded_passthrough_is_written_at_its_playback_level(speech):
    config = Config()
    config.render.recorded_live = "passthrough"
    live, device = UtteranceRenderer(config).render(0.05 * speech, SR, "rec", 1, origin="recorded")
    assert live.meta["source"]["passthrough"] is True and live.meta["origin"] == "recorded"
    assert len(live.audio) == len(device.audio)
    assert active_level_db(live.audio, live.sr, weighted=False) == pytest.approx(
        active_level_db(device.audio, device.sr, weighted=False), abs=0.5
    )


def test_impossible_combinations_are_refused(speech):
    renderer = UtteranceRenderer(Config())
    with pytest.raises(ValueError, match="rerecorded"):
        renderer.render(speech, SR, "rec", 0, origin="recorded", playback="replay")
    config = Config()
    config.render.mode = "live"
    with pytest.raises(ValueError, match="passthrough"):
        UtteranceRenderer(config).render(speech, SR, "rec", 0, origin="recorded")
    with pytest.raises(ValueError, match="origin"):
        renderer.render(speech, SR, "utt", 0, origin="studio")


# -- config ---------------------------------------------------------------------


def test_parse_forms_reads_every_spelling():
    assert parse_forms("replay") == {"replay": 1.0}
    assert parse_forms("single=2, replay=1") == {"single": 2.0, "replay": 1.0}
    assert parse_forms("single,rerecorded") == {"single": 1.0, "rerecorded": 1.0}
    assert parse_forms(["replay"]) == {"replay": 1.0}
    assert parse_forms({"single": 3}) == {"single": 3.0}
    with pytest.raises(ValueError, match="unknown playback form"):
        parse_forms("echo")
    with pytest.raises(ValueError, match="positive"):
        parse_forms({"single": 0})


def test_playback_config_is_validated():
    with pytest.raises(ValueError, match="unknown playback form"):
        Config.from_dict({"playback": {"forms": {"echo": 1.0}}}).validate()
    with pytest.raises(ValueError, match="capture key"):
        Config.from_dict({"playback": {"replay": {"capture": {"gain": 3}}}}).validate()
    with pytest.raises(ValueError, match="unknown codec"):
        Config.from_dict({"playback": {"replay": {"codecs": {"flac": 1.0}}}}).validate()
    with pytest.raises(ValueError, match="unknown environment"):
        Config.from_dict({"playback": {"rerecorded": {"environments": {"garage": 1}}}}).validate()
    with pytest.raises(ValueError, match=r"render\.origin"):
        Config.from_dict({"render": {"origin": "studio"}}).validate()
    with pytest.raises(ValueError, match="measured_mode"):
        Config.from_dict({"room": {"measured_mode": "half"}}).validate()


# -- scenes and mixes -------------------------------------------------------------


@pytest.fixture
def pool(tmp_path) -> list[Utterance]:
    out = []
    for i in range(4):
        path = tmp_path / f"u{i}.wav"
        sf.write(path, speech_like(seconds=1.0 + 0.2 * i, seed=i, f0=120 + 25 * i), SR)
        out.append(Utterance(Path(path), f"u{i}.wav", f"text {i}", f"spk{i % 2}"))
    return out


def _scene_config(**forms) -> Config:
    config = Config()
    config.scene.duration_s = [5.0, 6.0]
    config.scene.types = {"both": 1.0}
    config.playback.forms = forms or {"single": 1.0}
    return config


@pytest.mark.parametrize("form", FORMS)
def test_scenes_play_programmes_in_every_form(form, pool):
    scene = SceneComposer(_scene_config(**{form: 1.0}), pool).compose(0)
    devices = [u for u in scene.utterances if u["role"] == "device"]
    assert devices and all(u["playback_form"] == form for u in devices)
    assert scene.meta["sources"]["device0"]["playback"]["form"] == form


def test_recorded_inputs_never_speak_as_the_live_user(pool):
    for u in pool[2:]:
        u.origin = "recorded"
    composer = SceneComposer(_scene_config(single=1.0, rerecorded=1.0, replay=1.0), pool)
    recorded_keys = {u.key for u in pool[2:]}
    for index in range(6):
        scene = composer.compose(index)
        for u in scene.utterances:
            if u["utt"] in recorded_keys:
                assert u["role"] == "device" and u["playback_form"] == "rerecorded"


def test_a_pool_of_recordings_still_makes_scenes(pool):
    for u in pool:
        u.origin = "recorded"
    scene = SceneComposer(_scene_config(replay=1.0), pool).compose(0)
    assert {u["role"] for u in scene.utterances} == {"device"}
    assert all(u["playback_form"] == "rerecorded" for u in scene.utterances)


def test_mix_plays_recorded_material_as_it_is(tmp_path):
    path = tmp_path / "rec.wav"
    sf.write(path, speech_like(seconds=0.6, seed=3), SR)
    config = Config()
    config.scene.duration_s = [0.6, 0.6]
    rendered = render_mic(playback=path, config=config, seed=2, playback_origin="recorded")
    playback = rendered.meta["sources"]["device0"]["playback"]
    assert playback["form"] == "rerecorded" and playback["origin"] == "recorded"
    assert all(s["playback_form"] == "rerecorded" for s in rendered.meta["segments"])
    with pytest.raises(ValueError, match="rerecorded"):
        render_mic(
            playback=path, config=config, playback_origin="recorded", playback_forms="single"
        )

    listing = tmp_path / "speech.jsonl"
    listing.write_text(json.dumps({"audio": "rec.wav", "origin": "recorded"}) + "\n", "utf-8")
    with pytest.raises(ValueError, match="dry speech"):
        mix_main(listing, None, output=tmp_path / "out", config=config)


# -- command line and manifests ------------------------------------------------------


def _jsonl_inputs(tmp_path) -> Path:
    lines = []
    for i, extra in enumerate(
        [{}, {"playback": "replay"}, {"origin": "recorded"}, {"playback": {"rerecorded": 1}}]
    ):
        path = tmp_path / "in" / f"u{i}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, speech_like(seconds=0.7, seed=i), SR)
        lines.append(json.dumps({"audio": f"in/u{i}.wav", "text": f"句子{i}", **extra}))
    listing = tmp_path / "inputs.jsonl"
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return listing


def test_jsonl_lines_carry_origin_and_playback(tmp_path):
    utts = read_utterances(_jsonl_inputs(tmp_path))
    assert [(u.origin, u.playback) for u in utts] == [
        (None, None),
        (None, "replay"),
        ("recorded", None),
        (None, {"rerecorded": 1}),
    ]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"audio": "in/u0.wav", "origin": "studio"}) + "\n", "utf-8")
    with pytest.raises(ValueError, match="origin must be"):
        read_utterances(bad)


def test_render_command_follows_each_input(tmp_path):
    out = tmp_path / "out"
    assert main(["render", "--input", str(_jsonl_inputs(tmp_path)), "--output", str(out),
                 "--workers", "1"]) == 0  # fmt: skip
    records = [json.loads(x) for x in (out / "manifest.jsonl").read_text("utf-8").splitlines()]
    by_key: dict[str, list[dict]] = {}
    for r in records:
        by_key.setdefault(r["key"], []).append(r)
    assert sorted(r["label"] for r in by_key["u0.wav"]) == ["device", "live"]
    assert [r["playback_form"] for r in by_key["u1.wav"] if r["label"] == "device"] == ["replay"]
    assert [(r["label"], r["origin"]) for r in by_key["u2.wav"]] == [("device", "recorded")]
    assert [r["playback_form"] for r in by_key["u3.wav"] if r["label"] == "device"] == [
        "rerecorded"
    ]

    audit = run_audit(out / "manifest.jsonl")
    assert set(audit["level_auc_by_playback_form"]) <= set(FORMS)
    assert "live" in audit["acoustic_sstd_db"]
    assert np.isfinite(audit["features"]["lead_s"]["auc_device_gt_live"])


def test_render_command_refuses_a_bad_line_before_starting(tmp_path, capsys):
    path = tmp_path / "u.wav"
    sf.write(path, speech_like(seconds=0.5), SR)
    listing = tmp_path / "in.jsonl"
    listing.write_text(json.dumps({"audio": "u.wav", "origin": "recorded", "playback": "single"})
                       + "\n", encoding="utf-8")  # fmt: skip
    assert main(["render", "--input", str(listing), "--output", str(tmp_path / "o")]) == 1
    assert "u.wav" in capsys.readouterr().err
    assert not (tmp_path / "o" / "manifest.jsonl").exists()


def test_forms_and_origin_flags(tmp_path):
    path = tmp_path / "u.wav"
    sf.write(path, speech_like(seconds=0.5), SR)
    (tmp_path / "in.txt").write_text("u.wav\n", encoding="utf-8")
    out = tmp_path / "o"
    assert main(["render", "--input", str(tmp_path / "in.txt"), "--output", str(out),
                 "--workers", "1", "--variants", "3", "--forms", "replay"]) == 0  # fmt: skip
    records = [json.loads(x) for x in (out / "manifest.jsonl").read_text("utf-8").splitlines()]
    assert {r["playback_form"] for r in records if r["label"] == "device"} == {"replay"}
    out2 = tmp_path / "o2"
    assert main(["render", "--input", str(tmp_path / "in.txt"), "--output", str(out2),
                 "--workers", "1", "--origin", "recorded"]) == 0  # fmt: skip
    records = [json.loads(x) for x in (out2 / "manifest.jsonl").read_text("utf-8").splitlines()]
    assert [(r["label"], r["origin"]) for r in records] == [("device", "recorded")]


# -- building blocks ---------------------------------------------------------------


def test_one_room_spreads_less_than_two_in_series():
    rng = np.random.default_rng(0)
    single = [spectral_std(synth_ir(SR, 0.6, -6.0, rng), SR) for _ in range(6)]
    double = [
        spectral_std(np.convolve(synth_ir(SR, 0.6, -6.0, rng), synth_ir(SR, 0.5, -6.0, rng)), SR)
        for _ in range(6)
    ]
    close = [spectral_std(synth_ir(SR, 0.6, 12.0, rng), SR) for _ in range(6)]
    assert 4.6 < np.median(single) < 6.6
    assert 7.0 < np.median(double) < 9.0
    assert np.median(close) < np.median(single) - 1.5


def test_room_tone_under_a_voice_does_not_change_its_speech_level(speech):
    rng = np.random.default_rng(4)
    tone = 10 ** (-28 / 20) * np.std(speech[speech != 0]) * rng.standard_normal(len(speech))
    assert speech_level_db(speech + tone, SR) == pytest.approx(speech_level_db(speech, SR), abs=0.3)
    assert active_level_db(speech + tone, SR) < speech_level_db(speech, SR) - 1.0


def test_noise_suppression_pulls_down_noise_and_keeps_the_voice():
    rng = np.random.default_rng(1)
    t = np.arange(4 * SR) / SR
    gate = (np.floor(t * 2) % 2 == 1).astype(float)
    tone = 0.1 * np.sin(2 * np.pi * 900 * t) * gate
    noise = 0.003 * rng.standard_normal(len(t))
    y, meta = suppress_noise(tone + noise, SR, max_db=12.0)
    assert meta["ns_applied"]
    quiet = gate == 0
    reduction = 10 * np.log10(np.mean(noise[quiet] ** 2) / np.mean(y[quiet] ** 2))
    assert reduction > 8.0
    kept = 10 * np.log10(np.mean(y[~quiet] ** 2) / np.mean((tone + noise)[~quiet] ** 2))
    assert abs(kept) < 1.0


def test_a_grafted_tail_keeps_directivity_distance_and_the_measured_decay():
    dims = np.array([5.0, 4.0, 2.7])
    room = Room(dims, absorption_for_rt60(dims, np.full(7, 0.35)))
    mic = np.array([3.5, 2.0, 1.2])
    measured = synth_ir(SR, 0.9, 0.0, np.random.default_rng(2))
    split = 0.05

    def grafted(src, axis, seed=3):
        cardioid = Directivity(np.array(axis), np.full(7, 0.5))
        rng = np.random.default_rng(seed)
        h, _ = room_impulse_response(room, np.array(src), mic, SR, rng, cardioid)
        return graft_tail(h, measured, SR, split)

    facing = grafted([2.5, 2.0, 1.2], [1.0, 0.0, 0.0])
    away = grafted([2.5, 2.0, 1.2], [-1.0, 0.0, 0.0])
    assert np.max(np.abs(facing)) > 2.0 * np.max(np.abs(away))
    near = grafted([2.8, 2.0, 1.2], [1.0, 0.0, 0.0])
    far = grafted([0.6, 0.6, 1.2], [1.0, 0.0, 0.0])
    assert estimate_drr(near, SR) > estimate_drr(far, SR) + 6.0
    assert estimate_t60(far, SR) > 0.6  # the measured room's decay, not the model's 0.35 s
