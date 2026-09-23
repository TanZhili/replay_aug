"""End to end through the command line, on a tiny input list."""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf
import yaml

from replay_opus.cli import main
from replay_opus.irbank import synth_ir

from .conftest import SR, speech_like


def _inputs(tmp_path) -> str:
    lines = []
    for i in range(2):
        path = tmp_path / "tts" / f"utt{i}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, speech_like(seconds=0.8, seed=i), SR)
        lines.append(f"tts/utt{i}.wav\t第{i}句话")
    listing = tmp_path / "inputs.txt"
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(listing)


def test_render_writes_paired_clips_and_a_manifest(tmp_path):
    out = tmp_path / "out"
    code = main(
        [
            "render",
            "--input",
            _inputs(tmp_path),
            "--output",
            str(out),
            "--workers",
            "1",
            "--variants",
            "2",
            "--set",
            "capture.agc_prob=0",
        ]
    )
    assert code == 0
    records = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 8
    assert {r["label"] for r in records} == {"live", "device"}
    assert all((out / r["audio"]).is_file() for r in records)
    assert (out / "live" / "utt0__v1.wav").is_file() and (out / "device" / "utt0__v1.wav").is_file()
    assert records[0]["text"].startswith("第")
    pairs = {}
    for r in records:
        pairs.setdefault(r["pair"], set()).add(r["label"])
    assert all(v == {"live", "device"} for v in pairs.values())

    code = main(
        [
            "audit",
            "--manifest",
            str(out / "manifest.jsonl"),
            "--output",
            str(tmp_path / "audit.json"),
        ]
    )
    assert code == 0
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["n_live"] == audit["n_device"] == 4
    assert audit["features"]["duration_s"]["auc_device_gt_live"] == 0.5


def test_flac_output(tmp_path):
    out = tmp_path / "out"
    code = main(
        [
            "render",
            "--input",
            _inputs(tmp_path),
            "--output",
            str(out),
            "--workers",
            "1",
            "--mode",
            "device",
            "--set",
            "output.format=flac",
        ]
    )
    assert code == 0
    record = json.loads((out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["audio"].endswith(".flac")
    audio, sr = sf.read(out / record["audio"])
    assert sr == 16_000 and len(audio) > 0


def test_scenes_write_mixtures_labels_and_stems(tmp_path):
    out = tmp_path / "scenes"
    code = main(
        [
            "scenes",
            "--input",
            _inputs(tmp_path),
            "--output",
            str(out),
            "--count",
            "2",
            "--workers",
            "1",
            "--stems",
            "--set",
            "scene.duration_s=[4, 5]",
        ]
    )
    assert code == 0
    records = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    for r in records:
        audio, sr = sf.read(out / r["audio"])
        assert sr == 16_000 and 4.0 <= len(audio) / sr <= 5.0
        assert all((out / p).is_file() for p in r.get("stems", {}).values())
        for s in r["segments"]:
            assert s["role"] in {"live", "device"}


def test_config_file_paths_resolve_relative_to_the_file(tmp_path, rng):
    noise = tmp_path / "cfg" / "noise" / "n.wav"
    noise.parent.mkdir(parents=True)
    sf.write(noise, 0.05 * rng.standard_normal(SR), SR)
    (tmp_path / "cfg" / "noise.txt").write_text("noise/n.wav\n", encoding="utf-8")
    config = tmp_path / "cfg" / "exp.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "environments": {"bedroom": 1.0},
                "noise": {"lists": {"home": "noise.txt"}},
                "render": {"mode": "live"},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert (
        main(
            [
                "render",
                "--input",
                _inputs(tmp_path),
                "--output",
                str(out),
                "--workers",
                "1",
                "--config",
                str(config),
            ]
        )
        == 0
    )
    record = json.loads((out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["scene"]["noise_source"] == "n.wav"
    assert record["label"] == "live"


def test_presets_and_ir_stats(tmp_path, rng, capsys):
    assert main(["presets", "--section", "devices", "--output", str(tmp_path / "p.yaml")]) == 0
    devices = yaml.safe_load((tmp_path / "p.yaml").read_text(encoding="utf-8"))["devices"]
    assert {"tv", "phone", "car_audio"} <= set(devices)

    ir = tmp_path / "irs" / "room.wav"
    ir.parent.mkdir()
    sf.write(ir, synth_ir(48_000, 0.3, 8.0, rng), 48_000)
    (tmp_path / "irs.txt").write_text("irs/room.wav\n", encoding="utf-8")
    assert (
        main(
            [
                "ir-stats",
                "--ir-list",
                str(tmp_path / "irs.txt"),
                "--output",
                str(tmp_path / "s.jsonl"),
            ]
        )
        == 0
    )
    row = json.loads((tmp_path / "s.jsonl").read_text(encoding="utf-8"))
    assert row["ir"] == "room.wav" and np.isfinite(row["t60_s"])


def test_bad_input_list_fails_cleanly(tmp_path, capsys):
    (tmp_path / "in.txt").write_text("nope.wav\n", encoding="utf-8")
    assert (
        main(["render", "--input", str(tmp_path / "in.txt"), "--output", str(tmp_path / "o")]) == 1
    )
    assert "do not exist" in capsys.readouterr().err
