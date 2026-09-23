"""Path lists replace directories as the way a run names its inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf

from replay_opus.pathlist import _disambiguate, read_path_list, read_utterances, relative_labels

SR = 32_000


def _write_wav(path, rng) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, 0.2 * rng.standard_normal(SR // 4), SR)


def test_read_path_list_skips_comments_quotes_and_duplicates(tmp_path, rng):
    audio = tmp_path / "a.wav"
    _write_wav(audio, rng)
    listing = tmp_path / "lists" / "inputs.txt"
    listing.parent.mkdir()
    listing.write_text('# comment\n\n../a.wav\n"../a.wav"\n', encoding="utf-8")
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


def test_utterance_lists_carry_text_and_speaker(tmp_path, rng):
    for name in ("a", "b", "c"):
        _write_wav(tmp_path / "tts" / f"{name}.wav", rng)
    listing = tmp_path / "inputs.txt"
    listing.write_text(
        "tts/a.wav\ntts/b.wav\t打开客厅的灯\ntts/c.wav\tspk7\tturn it\tdown\n",
        encoding="utf-8",
    )
    utts = read_utterances(listing)
    assert [u.key for u in utts] == ["a.wav", "b.wav", "c.wav"]
    assert (utts[0].text, utts[0].speaker) == (None, None)
    assert (utts[1].text, utts[1].speaker) == ("打开客厅的灯", None)
    assert (utts[2].text, utts[2].speaker) == ("turn it\tdown", "spk7")
    # A list with text columns still reads as a plain path list.
    assert len(read_path_list(listing, suffixes=(".wav",))) == 3


def test_utterance_lists_in_jsonl(tmp_path, rng):
    _write_wav(tmp_path / "x" / "a.wav", rng)
    _write_wav(tmp_path / "x" / "b.wav", rng)
    listing = tmp_path / "inputs.jsonl"
    listing.write_text(
        json.dumps({"audio": "x/a.wav", "text": "你好", "speaker": 3}, ensure_ascii=False)
        + "\n"
        + json.dumps({"path": "x/b.wav"})
        + "\n",
        encoding="utf-8",
    )
    utts = read_utterances(listing)
    assert [(u.key, u.text, u.speaker) for u in utts] == [
        ("a.wav", "你好", "3"),
        ("b.wav", None, None),
    ]
    assert not list(tmp_path.glob(".*"))  # nothing staged next to the list

    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_utterances(broken)


def test_utterance_list_errors_name_the_line(tmp_path):
    listing = tmp_path / "inputs.txt"
    listing.write_text("missing.wav\ttext\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="line 1"):
        read_utterances(listing)
