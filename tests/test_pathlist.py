"""Path lists replace directories as the way a run names its inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from replay_opus.chain import ReplayChain
from replay_opus.cli import main
from replay_opus.config import Config
from replay_opus.irbank import synth_ir
from replay_opus.noise import NoiseBank
from replay_opus.pathlist import _disambiguate, read_path_list, relative_labels
from replay_opus.speaker import synth_speaker_kernels

SR = 48_000


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _write_wav(path, rng) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, 0.2 * rng.standard_normal(SR // 4), SR)


def test_read_path_list_skips_comments_quotes_and_duplicates(tmp_path, rng):
    audio = tmp_path / "a.wav"
    _write_wav(audio, rng)
    listing = tmp_path / "lists" / "inputs.txt"
    listing.parent.mkdir()
    listing.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "../a.wav",
                '"../a.wav"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    got = read_path_list(listing, kind="input audio", suffixes=(".wav",))
    assert len(got) == 1
    assert got[0].samefile(audio)


def test_read_path_list_accepts_utf16(tmp_path, rng):
    audio = tmp_path / "a.wav"
    _write_wav(audio, rng)
    listing = tmp_path / "inputs.txt"
    listing.write_text(f"{audio.name}\n", encoding="utf-16")
    got = read_path_list(listing, suffixes=(".wav",))
    assert len(got) == 1 and got[0].samefile(audio)


def test_read_path_list_rejects_a_directory_missing_files_and_bad_suffixes(tmp_path):
    with pytest.raises(ValueError, match="directory"):
        read_path_list(tmp_path, kind="input audio")

    listing = tmp_path / "missing.txt"
    listing.write_text("nope.wav\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="do not exist"):
        read_path_list(listing)

    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("notes.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported suffix"):
        read_path_list(bad, suffixes=(".wav",))

    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no "):
        read_path_list(empty)


def test_relative_labels_keep_the_subtree():
    labels = relative_labels([Path("data/set_a/x.wav"), Path("data/set_b/x.wav")])
    assert labels == ["set_a/x.wav", "set_b/x.wav"]


def test_disambiguate_repeated_names():
    assert _disambiguate(["a.wav", "a.wav", "b.wav"]) == ["a.wav", "a__2.wav", "b.wav"]


def test_measured_room_noise_and_speaker_come_from_lists(tmp_path, rng):
    ir_path = tmp_path / "irs" / "room.wav"
    ir_path.parent.mkdir()
    sf.write(ir_path, synth_ir(SR, 0.3, 8.0, rng), SR)
    ir_list = tmp_path / "irs.txt"
    ir_list.write_text("irs/room.wav\n", encoding="utf-8")

    noise_path = tmp_path / "noise" / "hvac.wav"
    noise_path.parent.mkdir()
    sf.write(noise_path, 0.01 * rng.standard_normal(SR), SR)
    noise_list = tmp_path / "noise.txt"
    noise_list.write_text("noise/hvac.wav\n", encoding="utf-8")

    model = synth_speaker_kernels(SR, rng, "low")
    model_path = tmp_path / "speakers" / "phone.npz"
    model_path.parent.mkdir()
    model.save(model_path)
    model_list = tmp_path / "speakers.txt"
    model_list.write_text("speakers/phone.npz\n", encoding="utf-8")

    config = Config()
    config.room.mode = "measured"
    config.room.ir_list = str(ir_list)
    config.noise.noise_list = str(noise_list)
    config.playback.model_list = str(model_list)
    chain = ReplayChain(config)
    meta = chain.process(0.2 * rng.standard_normal(SR // 2), SR, key="a", variant=0).meta
    assert meta["room_ir"] == "room.wav"
    assert meta["noise"] == "hvac.wav"
    assert meta["speaker"] == "synthetic:low"
    assert "room_ir_measured" in meta


def test_noise_bank_without_a_list_stays_synthetic(rng):
    noise, label = NoiseBank.from_list(None, SR).sample(SR // 10, rng)
    assert label.startswith("synthetic:")
    assert len(noise) == SR // 10


def test_cli_simulate_and_ir_stats_read_path_lists(tmp_path, rng):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    sf.write(src_dir / "utt.wav", 0.2 * rng.standard_normal(SR // 5), SR)
    inputs = tmp_path / "inputs.txt"
    inputs.write_text("in/utt.wav\n", encoding="utf-8")

    ir_path = tmp_path / "irs" / "room.wav"
    ir_path.parent.mkdir()
    sf.write(ir_path, synth_ir(SR, 0.25, 10.0, rng), SR)
    ir_list = tmp_path / "irs.txt"
    ir_list.write_text("irs/room.wav\n", encoding="utf-8")

    out = tmp_path / "out"
    code = main(
        [
            "simulate",
            "--input",
            str(inputs),
            "--output",
            str(out),
            "--room-ir-list",
            str(ir_list),
            "--room-mode",
            "measured",
            "--workers",
            "1",
            "--variants",
            "1",
        ]
    )
    assert code == 0
    assert (out / "utt.wav").is_file()
    record = json.loads((out / "manifest.jsonl").read_text(encoding="utf-8"))
    assert record["key"] == "utt.wav"
    assert record["room_ir"] == "room.wav"

    stats_path = tmp_path / "stats.jsonl"
    code = main(["ir-stats", "--ir-list", str(ir_list), "--output", str(stats_path)])
    assert code == 0
    row = json.loads(stats_path.read_text(encoding="utf-8"))
    assert row["ir"] == "room.wav"
    assert {"t60_s", "drr_db", "c50_db", "length_ms"} <= set(row)
    assert np.isfinite(row["t60_s"])
