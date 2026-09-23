"""Third-party replay detectors, used as independent judges of rendered audio.

Optional: needs ``torch`` (and ``transformers`` for the wav2vec2 judge), installed
with ``pip install -r requirements-detect.txt``. Weights are downloaded on first use
into ``~/.cache/replay_opus/detectors`` (override with ``REPLAY_OPUS_CACHE``).

Every detector returns a *replay score*: higher means more likely replayed. Both
checkpoints here come from the ASVspoof physical-access task, whose bona fide class
is speech played once into a room by a flat, high-quality loudspeaker and whose
spoof class is that speech recorded by an attacker and replayed through another
device - two acoustic passes. They were trained to spot the second pass, not to
tell a mouth from a loudspeaker, which is what the controls in ``judge.py`` are for.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dsp import active_level_db, resample_to

SR = 16_000


def cache_dir() -> Path:
    root = Path(os.environ.get("REPLAY_OPUS_CACHE", Path.home() / ".cache" / "replay_opus"))
    path = root / "detectors"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _torch():
    try:
        import torch  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the replay detectors need PyTorch: pip install -r requirements-detect.txt"
        ) from exc
    return torch


def prepare(x: np.ndarray, sr: int, level_dbfs: float | None = -26.0) -> np.ndarray:
    """16 kHz mono float32, optionally set to a fixed active level.

    Levelling keeps the judges from reacting to how loud a file happens to be: the
    LCNN's first cepstral coefficient is frame energy.
    """
    y = resample_to(np.asarray(x, dtype=np.float64), sr, SR)
    if level_dbfs is not None and np.any(y):
        y = y * 10.0 ** ((level_dbfs - active_level_db(y, SR, weighted=False)) / 20.0)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# ASVspoof 2021 PA baseline: LFCC front end + LCNN/BLSTM back end
# ---------------------------------------------------------------------------

LCNN_URLS = (
    "https://www.asvspoof.org/asvspoof2021/pre_trained_PA_LFCC-LCNN.zip",
    "https://www.dropbox.com/sh/gf3zp00qvdp3row/AACOBr0ymqsMA0rKctMxkKaxa/temp/asvspoof2021/"
    "pa_trained_network.pt?dl=1",
)


def _fetch_lcnn() -> Path:
    target = cache_dir() / "asvspoof2021_pa_lcnn" / "pa_trained_network.pt"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in LCNN_URLS:
        download = target.parent / Path(url.split("?")[0]).name
        try:
            urllib.request.urlretrieve(url, download)  # noqa: S310 - fixed https URLs
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            continue
        if download.suffix == ".zip":
            with zipfile.ZipFile(download) as archive:
                archive.extract("pa_trained_network.pt", target.parent)
            download.unlink()
        if target.is_file():
            return target
    raise RuntimeError("could not download the ASVspoof 2021 PA LCNN:\n" + "\n".join(errors))


def _build_lcnn(torch):
    """The official ``baseline_PA`` network, rewritten against the current torch API.

    Module names mirror the original so the published state dict loads unchanged; the
    original code relies on ``torch.rfft`` and the pre-1.8 ``torch.stft`` output.
    """
    nn = torch.nn

    class MaxFeatureMap2D(nn.Module):
        def forward(self, x):
            b, c = x.shape[:2]
            return x.view(b, 2, c // 2, *x.shape[2:]).max(1).values

    class BLSTMLayer(nn.Module):
        def __init__(self, dim_in: int, dim_out: int):
            super().__init__()
            self.l_blstm = nn.LSTM(dim_in, dim_out // 2, bidirectional=True)

        def forward(self, x):
            return self.l_blstm(x.permute(1, 0, 2))[0].permute(1, 0, 2)

    class LFCC(nn.Module):
        """20 linear-filterbank cepstra over 0-4 kHz, c0 replaced by log energy, + deltas."""

        def __init__(self):
            super().__init__()
            self.lfcc_fb = nn.Parameter(torch.zeros(256, 20), requires_grad=False)
            self.l_dct = nn.Linear(20, 20, bias=False)
            self.register_buffer("window", torch.hamming_window(320), persistent=False)

        @staticmethod
        def delta(x):
            padded = nn.functional.pad(x.unsqueeze(1), (0, 0, 1, 1), "replicate").squeeze(1)
            return padded[:, 2:] - padded[:, : x.shape[1]]

        def forward(self, x):
            x = torch.cat([x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]], dim=1)
            spec = torch.stft(
                x,
                1024,
                160,
                320,
                window=self.window,
                onesided=True,
                pad_mode="constant",
                return_complex=True,
            )
            power = spec.abs().pow(2).permute(0, 2, 1)[:, :, :256]
            eps = torch.finfo(torch.float32).eps
            lfcc = self.l_dct(torch.log10(power @ self.lfcc_fb + eps))
            lfcc[:, :, 0] = torch.log10((power / 1024).sum(dim=2) + eps)
            d1 = self.delta(lfcc)
            return torch.cat([lfcc, d1, self.delta(d1)], dim=2)

    def block(c_in, c_mid, c_out):
        """1x1 conv + MFM + BN, then 3x3 conv + MFM: the repeated LCNN unit."""
        return [
            nn.Conv2d(c_in, c_mid, 1),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(c_mid // 2, affine=False),
            nn.Conv2d(c_mid // 2, c_out, 3, padding=1),
            MaxFeatureMap2D(),
        ]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("input_mean", "input_std", "output_mean", "output_std"):
                self.register_parameter(name, nn.Parameter(torch.zeros(1), requires_grad=False))
            self.m_frontend = nn.ModuleList([LFCC()])
            self.m_transform = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(1, 64, 5, padding=2),
                        MaxFeatureMap2D(),
                        nn.MaxPool2d(2, 2),
                        *block(32, 64, 96),
                        nn.MaxPool2d(2, 2),
                        nn.BatchNorm2d(48, affine=False),
                        *block(48, 96, 128),
                        nn.MaxPool2d(2, 2),
                        *block(64, 128, 64),
                        nn.BatchNorm2d(32, affine=False),
                        *block(32, 64, 64),
                        nn.MaxPool2d(2, 2),
                        nn.Dropout(0.7),
                    )
                ]
            )
            self.m_before_pooling = nn.ModuleList(
                [nn.Sequential(BLSTMLayer(96, 96), BLSTMLayer(96, 96))]
            )
            self.m_output_act = nn.ModuleList([nn.Linear(96, 1)])

        def forward(self, wav):
            feats = self.m_frontend[0](wav)
            hidden = self.m_transform[0](feats.unsqueeze(1))
            hidden = hidden.permute(0, 2, 1, 3).reshape(wav.shape[0], hidden.shape[2], -1)
            pooled = (self.m_before_pooling[0](hidden) + hidden).mean(1)
            return self.m_output_act[0](pooled).squeeze(1)  # bona fide logit

    return Model()


@dataclass
class LCNNDetector:
    """Official ASVspoof 2021 PA baseline (LFCC-LCNN), trained on ASVspoof 2019 PA."""

    name: str = "asvspoof21-pa-lcnn"
    title: str = (
        "ASVspoof 2021 PA baseline LFCC-LCNN (official weights, trained on simulated 2019 PA)"
    )

    def __post_init__(self) -> None:
        torch = _torch()
        self._torch = torch
        self.model = _build_lcnn(torch)
        state = torch.load(_fetch_lcnn(), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def score(self, x16: np.ndarray) -> float:
        with self._torch.no_grad():
            logit = self.model(self._torch.from_numpy(x16)[None, :])
        return float(-logit[0])


@dataclass
class Wav2Vec2PADetector:
    """wav2vec2-base fine-tuned on real ASVspoof 2021 PA recordings (community model)."""

    name: str = "w2v2-pa21"
    title: str = (
        "wav2vec2-base fine-tuned on ASVspoof 2021 PA real replay "
        "(Vansh180/deepfake-audio-wav2vec2)"
    )
    repo: str = "Vansh180/deepfake-audio-wav2vec2"
    revision: str = "e1197e2063deb056b0a6a348f3913d27de7f3d83"

    def __post_init__(self) -> None:
        torch = _torch()
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the wav2vec2 judge needs transformers: pip install -r requirements-detect.txt"
            ) from exc
        path = snapshot_download(
            self.repo, revision=self.revision, cache_dir=str(cache_dir() / "hf")
        )
        self._torch = torch
        self.extractor = AutoFeatureExtractor.from_pretrained(path)
        self.model = AutoModelForAudioClassification.from_pretrained(path).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self._fake, self._real = labels["fake"], labels["real"]

    def score(self, x16: np.ndarray) -> float:
        inputs = self.extractor(x16, sampling_rate=SR, return_tensors="pt")
        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0]
        return float(logits[self._fake] - logits[self._real])


DETECTORS = {
    LCNNDetector.name: LCNNDetector,
    Wav2Vec2PADetector.name: Wav2Vec2PADetector,
}


def load_detector(name: str):
    if name not in DETECTORS:
        raise ValueError(f"unknown detector {name!r}; choose from {sorted(DETECTORS)}")
    return DETECTORS[name]()


__all__ = ["DETECTORS", "LCNNDetector", "Wav2Vec2PADetector", "load_detector", "prepare"]
