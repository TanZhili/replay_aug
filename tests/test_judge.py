"""Judging rendered audio with replay detectors: the harness, the controls, the port."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
import soundfile as sf
import yaml

from replay_opus import detectors, judge
from replay_opus.cli import main
from replay_opus.config import Config
from replay_opus.detectors import SR, cache_dir, prepare
from replay_opus.dsp import active_level_db
from replay_opus.render import UtteranceRenderer
from replay_opus.scene import derive_seed

from .conftest import speech_like


@dataclass
class HighBandDetector:
    """Stand-in judge: share of energy above 3 kHz. Loudspeakers and codecs move it."""

    name: str = "high-band"
    title: str = "fraction of energy above 3 kHz"

    def score(self, x16: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(x16)) ** 2
        freqs = np.fft.rfftfreq(len(x16), 1 / SR)
        return float(spectrum[freqs > 3000].sum() / (spectrum.sum() + 1e-20))


@pytest.fixture
def rendered(tmp_path):
    for i in range(3):
        sf.write(tmp_path / f"u{i}.wav", speech_like(seconds=1.0, seed=i), 32_000)
    (tmp_path / "in.txt").write_text("u0.wav\nu1.wav\nu2.wav\n", encoding="utf-8")
    out = tmp_path / "out"
    assert (
        main(
            ["render", "--input", str(tmp_path / "in.txt"), "--output", str(out), "--workers", "1"]
        )
        == 0
    )
    return out


def test_prepare_sets_rate_and_level():
    x = 0.01 * speech_like(sr=24_000)
    y = prepare(x, 24_000, level_dbfs=-26.0)
    assert y.dtype == np.float32
    assert len(y) == pytest.approx(len(x) * 2 / 3, abs=2)
    assert active_level_db(y.astype(np.float64), SR, weighted=False) == pytest.approx(
        -26.0, abs=0.1
    )


def test_wilson_interval_brackets_the_proportion():
    lo, hi = judge.wilson(9, 12)
    assert lo < 0.75 < hi
    assert judge.wilson(12, 12)[1] == pytest.approx(1.0)


def test_render_saves_a_config_that_rebuilds_the_scenes(rendered):
    saved = yaml.safe_load((rendered / "config.yaml").read_text(encoding="utf-8"))
    assert Config.from_dict(saved).to_dict() == Config().to_dict()


def test_replay_control_uses_the_same_scene_and_device(rendered):
    records = [
        json.loads(line)
        for line in (rendered / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    device = next(r for r in records if r["label"] == "device")
    renderer = UtteranceRenderer(Config())
    audio, sr = judge._replay_control(renderer, device)
    assert sr == 16_000
    assert len(audio) == pytest.approx(device["duration_s"] * sr, abs=2)

    scene = renderer.scenes.sample_scene(
        derive_seed(1234, device["key"], device["variant"]), device["key"]
    )
    raw = renderer.scenes.render_device(scene, np.ones(32_000) * 0.01, raw_programme=True)
    assert raw.meta["drivers"] == device["source"]["drivers"]  # same loudspeaker, same spot
    assert raw.meta["content"] == {"content": "raw"}


def test_replay_control_refuses_a_different_config(rendered):
    records = [
        json.loads(line)
        for line in (rendered / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    other = Config()
    other.seed = 99
    with pytest.raises(ValueError, match="different room"):
        judge._replay_control(UtteranceRenderer(other), records[0])


def test_judge_reports_paired_statistics(rendered, monkeypatch):
    monkeypatch.setitem(detectors.DETECTORS, "high-band", HighBandDetector)
    result = judge.run_judge(rendered / "manifest.jsonl", ["high-band"])
    entry = result["detectors"]["high-band"]
    assert result["n_pairs"] == 3
    assert set(entry["median"]) == set(judge.CONDITIONS)
    assert {"device>live", "replay>live", "live>dry", "device>dry"} == set(entry["auc"])
    paired = entry["paired"]["device>live"]
    assert paired["n"] == 3 and 0.0 <= paired["win_rate"] <= 1.0
    assert sum(v["n"] for v in entry["by_playback_form"].values()) == 3
    assert set(entry["by_playback_form"]) <= {"single", "rerecorded", "replay"}
    assert all(set(s) == set(judge.CONDITIONS) for s in entry["scores"].values())
    assert "same scene" in judge.format_judge_report(result)


def test_judge_scores_a_reference_list(rendered, monkeypatch, tmp_path):
    monkeypatch.setitem(detectors.DETECTORS, "high-band", HighBandDetector)
    lines = []
    for label in ("live", "device"):
        for path in sorted((rendered / label).glob("*.wav")):
            lines.append(f"{path.as_posix()}\t{'bonafide' if label == 'live' else 'spoof'}")
    (tmp_path / "ref.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = judge.run_judge(
        rendered / "manifest.jsonl", ["high-band"], controls=False, reference=tmp_path / "ref.txt"
    )
    reference = result["detectors"]["high-band"]["reference"]
    assert reference["n_spoof"] == reference["n_bonafide"] == 3


def test_judge_command_writes_scores(rendered, monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(detectors.DETECTORS, "high-band", HighBandDetector)
    out = tmp_path / "judge.json"
    code = main(
        [
            "judge",
            "--manifest",
            str(rendered / "manifest.jsonl"),
            "--detector",
            "high-band",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert "replay>live" in capsys.readouterr().out
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["detectors"]["high-band"]["paired"]["replay>live"]["n"] == 3


def test_judge_without_a_saved_config_needs_one(rendered):
    (rendered / "config.yaml").unlink()
    with pytest.raises(ValueError, match=r"config\.yaml not found"):
        judge.run_judge(rendered / "manifest.jsonl", ["high-band"])


def test_lcnn_front_end_shapes():
    torch = pytest.importorskip("torch")
    model = detectors._build_lcnn(torch)
    feats = model.m_frontend[0](torch.zeros(2, 16_000) + 1e-3)
    assert tuple(feats.shape) == (2, 101, 60)


def test_official_lcnn_weights_load_into_the_port():
    torch = pytest.importorskip("torch")
    weights = cache_dir() / "asvspoof2021_pa_lcnn" / "pa_trained_network.pt"
    if not weights.is_file():
        pytest.skip("official weights not downloaded yet (run `replay_opus judge` once)")
    model = detectors._build_lcnn(torch)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    x = torch.from_numpy(prepare(speech_like(sr=16_000, seconds=2.0), 16_000))[None, :]
    with torch.no_grad():
        assert torch.isfinite(model(x)).all()
