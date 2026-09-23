"""Microphone mixes: user speech and device playback, with optional noise and a measured room."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from replay_opus.cli import main as cli_main
from replay_opus.config import Config
from replay_opus.irbank import synth_ir
from replay_opus.mix import main, render_mic
from replay_opus.scene import SceneRenderer

from .conftest import SR, speech_like


def _config() -> Config:
    config = Config()
    config.capture.agc_prob = 0
    config.capture.uplink_prob = 0
    config.scene.duration_s = [0.6, 0.6]
    return config


def _write_speech(path, seed: int) -> None:
    sf.write(path, speech_like(seconds=0.45, seed=seed), SR)


@pytest.fixture
def scenes():
    return SceneRenderer(_config())


def test_every_input_can_be_omitted(scenes):
    rendered = render_mic(scenes=scenes, seed=1)
    assert rendered.sr == 16_000
    assert rendered.audio.ndim == 1
    assert len(rendered.audio) > 0
    assert np.all(np.isfinite(rendered.audio))
    assert rendered.meta["inputs"] == {
        "speech": None,
        "playback": None,
        "noise": None,
        "rir": None,
    }
    assert rendered.meta["segments"] == []
    assert rendered.meta["scene"]["noise_source"].startswith("synthetic:")
    again = render_mic(scenes=scenes, seed=1)
    assert np.array_equal(rendered.audio, again.audio)


def test_user_and_device_are_both_in_the_microphone(tmp_path, scenes):
    user = tmp_path / "user.wav"
    device = tmp_path / "device.wav"
    _write_speech(user, 1)
    _write_speech(device, 2)
    live = render_mic(speech=user, scenes=scenes, seed=4)
    played = render_mic(playback=device, scenes=scenes, seed=4)
    both = render_mic(speech=user, playback=device, scenes=scenes, seed=4)
    assert not np.allclose(both.audio, live.audio)
    assert not np.allclose(both.audio, played.audio)
    assert {s["role"] for s in both.meta["segments"]} == {"live", "device"}
    assert set(both.meta["sources"]) == {"live0", "device0"}
    assert both.meta["inputs"]["speech"].endswith("user.wav")
    assert both.meta["inputs"]["playback"].endswith("device.wav")

    config = _config()
    config.noise.enabled = False
    quiet = SceneRenderer(config)
    only_user = render_mic(speech=user, scenes=quiet, seed=5)
    mask = np.zeros(len(only_user.audio), dtype=bool)
    for segment in only_user.meta["segments"]:
        if segment["role"] == "live":
            a = int(segment["start"] * only_user.sr)
            b = int(segment["end"] * only_user.sr)
            mask[a:b] = True
    assert mask.any() and (~mask).any()
    inside = np.mean(only_user.audio[mask] ** 2)
    outside = np.mean(only_user.audio[~mask] ** 2)
    assert 10 * np.log10(inside / (outside + 1e-20)) > 10.0


def test_same_file_can_be_user_and_playback(tmp_path, scenes):
    utt = tmp_path / "utt.wav"
    _write_speech(utt, 3)
    rendered = render_mic(speech=utt, playback=utt, scenes=scenes, seed=6)
    assert rendered.meta["inputs"]["speech"] == rendered.meta["inputs"]["playback"]
    assert {s["role"] for s in rendered.meta["segments"]} == {"live", "device"}


def test_noise_file_and_measured_rir_replace_the_models(tmp_path, scenes):
    utt = tmp_path / "utt.wav"
    _write_speech(utt, 4)
    noise = tmp_path / "hvac.wav"
    sf.write(noise, np.random.default_rng(0).standard_normal(SR), SR)
    rir = tmp_path / "living.wav"
    sf.write(rir, synth_ir(SR, t60=0.25, drr=6.0, rng=np.random.default_rng(1)), SR)

    plain = render_mic(speech=utt, scenes=scenes, seed=7)
    with_noise = render_mic(speech=utt, noise=noise, scenes=scenes, seed=7)
    with_rir = render_mic(speech=utt, rir=rir, scenes=scenes, seed=7)
    assert not np.allclose(plain.audio, with_noise.audio)
    assert not np.allclose(plain.audio, with_rir.audio)
    assert with_noise.meta["scene"]["noise_source"] == "hvac.wav"
    assert with_rir.meta["scene"]["measured_room"] == "living"
    assert with_rir.meta["inputs"]["rir"].endswith("living.wav")
    assert "measured_room" not in plain.meta["scene"]


def test_missing_file_is_rejected(tmp_path, scenes):
    with pytest.raises(FileNotFoundError):
        render_mic(speech=tmp_path / "missing.wav", scenes=scenes)
    with pytest.raises(ValueError, match="directory"):
        render_mic(rir=tmp_path, scenes=scenes)
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not an ir", encoding="utf-8")
    with pytest.raises(ValueError, match="impulse response"):
        render_mic(rir=bogus, scenes=scenes)


def test_main_aligns_optional_lists_and_cycles_the_short_ones(tmp_path):
    speech_lines = []
    for i in range(2):
        path = tmp_path / "tts" / f"u{i}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_speech(path, i)
        speech_lines.append(f"tts/u{i}.wav\t第{i}句")
    speech_list = tmp_path / "speech.txt"
    speech_list.write_text("\n".join(speech_lines) + "\n", encoding="utf-8")
    rir = tmp_path / "room.wav"
    sf.write(rir, synth_ir(SR, t60=0.2, drr=4.0, rng=np.random.default_rng(2)), SR)
    rir_list = tmp_path / "rirs.txt"
    rir_list.write_text("room.wav\n", encoding="utf-8")

    out = tmp_path / "out"
    code = main(
        speech_list,
        None,
        None,
        rir_list,
        out,
        config=_config(),
        workers=1,
    )
    assert code == 0
    text = (out / "manifest.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines()]
    assert [r["audio"] for r in records] == ["mix_000000.wav", "mix_000001.wav"]
    assert [r["text"] for r in records] == ["第0句", "第1句"]
    assert records[0]["inputs"]["playback"] is None
    assert records[0]["inputs"]["rir"] == records[1]["inputs"]["rir"]
    assert records[0]["inputs"]["speech"] != records[1]["inputs"]["speech"]
    assert all(r["scene"]["measured_room"] == "room" for r in records)
    assert all((out / r["audio"]).is_file() for r in records)
    assert {s["role"] for s in records[0]["segments"]} == {"live"}

    same = tmp_path / "same"
    assert main(speech_list, speech_list, output=same, config=_config(), workers=1) == 0
    paired = [
        json.loads(line)
        for line in (same / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert paired[0]["inputs"]["speech"] == paired[0]["inputs"]["playback"]
    assert paired[0]["playback_text"] == "第0句"
    assert {s["role"] for s in paired[0]["segments"]} == {"live", "device"}


def test_main_requires_a_list_or_a_count(tmp_path):
    with pytest.raises(ValueError, match="count"):
        main(output=tmp_path / "empty")
    out = tmp_path / "one"
    assert main(output=out, config=_config(), count=1, workers=1) == 0
    record = json.loads((out / "manifest.jsonl").read_text(encoding="utf-8"))
    assert record["inputs"]["speech"] is None
    assert (out / "mix_000000.wav").is_file()


def test_parallel_workers_write_every_mix(tmp_path):
    utt = tmp_path / "a.wav"
    _write_speech(utt, 9)
    listing = tmp_path / "a.txt"
    listing.write_text("a.wav\n", encoding="utf-8")
    out = tmp_path / "parallel"
    assert main(listing, output=out, config=_config(), count=2, workers=2) == 0
    names = sorted(p.name for p in out.glob("mix_*.wav"))
    assert names == ["mix_000000.wav", "mix_000001.wav"]
    records = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert all("error" not in record for record in records)


def test_cli_mix(tmp_path):
    utt = tmp_path / "a.wav"
    _write_speech(utt, 8)
    listing = tmp_path / "a.txt"
    listing.write_text("a.wav\t开灯\n", encoding="utf-8")
    out = tmp_path / "cli"
    code = cli_main(
        [
            "mix",
            "--speech-list",
            str(listing),
            "--playback-list",
            str(listing),
            "--output",
            str(out),
            "--workers",
            "1",
            "--set",
            "scene.duration_s=[0.5, 0.5]",
            "--set",
            "capture.agc_prob=0",
        ]
    )
    assert code == 0
    record = json.loads((out / "manifest.jsonl").read_text(encoding="utf-8"))
    assert record["text"] == "开灯"
    assert record["inputs"]["speech"] == record["inputs"]["playback"]
    assert (out / "config.yaml").is_file()
