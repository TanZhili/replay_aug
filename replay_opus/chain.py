"""The five-stage replay chain: playback, room, noise, capture, transport.

Each generated variant draws its own configuration from the ranges in
:class:`~replay_opus.config.Config` using a per-variant RNG, so a given
``(seed, input path, variant index)`` always reproduces the same output and the
run can be resumed or sharded without duplicating conditions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .capture import adc_clip, apply_mic_response, simple_agc
from .config import Config
from .dsp import fade_edges, resample_to, rms_db, sample_range, set_peak_db, to_mono
from .irbank import IRBank, reshape_ir, simulate_room_ir, synth_ir
from .metrics import describe_ir
from .noise import NoiseBank, mix_at_snr
from .pathlist import read_path_list
from .speaker import SpeakerModel, apply_speaker, synth_speaker_kernels
from .transport import CodecSpec, apply_transport, available_codecs


def variant_seed(base_seed: int, key: str, variant: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}:{variant}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class ChainResult:
    audio: np.ndarray
    sr: int
    meta: dict


class ReplayChain:
    """Applies the replay chain to audio. Construct once, call many times."""

    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.sr = config.sr_internal

        room = config.room
        self.ir_bank = (
            IRBank.from_list(room.ir_list, self.sr, room.ir_max_ms)
            if room.mode in {"measured", "reshaped"}
            else None
        )
        self.mic_bank = (
            IRBank.from_list(config.capture.mic_ir_list, self.sr, 200.0)
            if config.capture.enabled and config.capture.mic_ir_list
            else None
        )
        self.noise_bank = (
            NoiseBank.from_list(config.noise.noise_list, self.sr) if config.noise.enabled else None
        )
        self.speaker_models = self._load_speaker_models()
        self.codec_names, self.codec_weights = self._resolve_codecs()

    def _load_speaker_models(self) -> list[SpeakerModel]:
        list_path = self.config.playback.model_list
        if not list_path:
            return []
        paths = read_path_list(list_path, kind="speaker model", suffixes=(".npz",))
        return [SpeakerModel.load(p, self.sr) for p in paths]

    def _resolve_codecs(self) -> tuple[list[str | None], np.ndarray]:
        usable = set(available_codecs())
        names, weights = [], []
        transport = self.config.transport
        for name, weight in zip(transport.codecs, transport.codec_weights, strict=True):
            if name is None or name in usable:
                names.append(name)
                weights.append(float(weight))
        if not names:
            names, weights = [None], [1.0]
        return names, np.asarray(weights) / sum(weights)

    def _room_ir(self, rng: np.random.Generator, meta: dict) -> np.ndarray:
        room = self.config.room
        t60 = sample_range(rng, room.t60_s)
        drr = sample_range(rng, room.drr_db)
        meta["room_mode"] = room.mode
        meta["room_target_t60_s"] = round(t60, 4)
        meta["room_target_drr_db"] = round(drr, 2)

        if room.mode == "synthetic":
            return synth_ir(self.sr, t60, drr, rng)
        if room.mode == "simulated":
            dims = [
                float(rng.uniform(lo, hi))
                for lo, hi in zip(room.room_dim_m, room.room_dim_max_m, strict=True)
            ]
            src, mic, distance = _place_source_and_mic(dims, room.distance_m, rng)
            meta["room_dim_m"] = [round(d, 2) for d in dims]
            meta["room_distance_m"] = round(distance, 3)
            return simulate_room_ir(self.sr, tuple(dims), tuple(src), tuple(mic), t60)

        assert self.ir_bank is not None
        ir, label = self.ir_bank.sample(rng)
        meta["room_ir"] = label
        if room.mode == "reshaped":
            ir = reshape_ir(ir, self.sr, t60, drr)
        return ir

    def process(self, audio: np.ndarray, sr_in: int, key: str, variant: int = 0) -> ChainResult:
        cfg = self.config
        rng = np.random.default_rng(variant_seed(cfg.seed, key, variant))
        meta: dict = {"key": key, "variant": variant, "sr_in": sr_in}

        x = resample_to(to_mono(audio), sr_in, self.sr)
        x = fade_edges(x, self.sr, 5.0)
        n_input = len(x)
        meta["input_level_dbfs"] = round(rms_db(x), 2)

        if cfg.playback.enabled:
            x, playback_meta = self._playback(x, rng)
            meta.update(playback_meta)

        if cfg.room.enabled:
            ir = self._room_ir(rng, meta)
            meta["room_ir_measured"] = describe_ir(ir, self.sr)
            x = signal.fftconvolve(x, ir)

        # Trim before the noise and capture stages so the noise floor and the AGC see
        # the same signal length the recorder would have seen.
        keep = n_input + int(self.sr * cfg.output.tail_ms / 1000.0)
        if len(x) > keep:
            x = fade_edges(x[:keep], self.sr, 10.0)
        meta["tail_ms"] = round(cfg.output.tail_ms, 1)

        if cfg.noise.enabled and self.noise_bank is not None:
            snr = sample_range(rng, cfg.noise.snr_db)
            noise, label = self.noise_bank.sample(len(x), rng)
            x = mix_at_snr(x, noise, snr)
            meta.update({"snr_db": round(snr, 2), "noise": label})

        if cfg.capture.enabled:
            x, capture_meta = self._capture(x, rng)
            meta.update(capture_meta)

        if cfg.transport.enabled:
            x, out_sr, transport_meta = self._transport(x, rng)
            meta.update(transport_meta)
        else:
            out_sr = cfg.output.sr
            x = resample_to(x, self.sr, out_sr)

        # The mic IR and the codec can each add their own tail, so enforce the length
        # contract once more at the output rate.
        keep_out = round(n_input * out_sr / self.sr) + int(out_sr * cfg.output.tail_ms / 1000.0)
        if len(x) > keep_out:
            x = fade_edges(x[:keep_out], out_sr, 10.0)

        peak = sample_range(rng, cfg.output.peak_db)
        x = set_peak_db(x, peak)
        meta["output_peak_db"] = round(peak, 2)
        meta["output_level_dbfs"] = round(rms_db(x), 2)
        meta["duration_s"] = round(len(x) / out_sr, 3)

        return ChainResult(audio=np.clip(x, -1.0, 1.0), sr=out_sr, meta=meta)

    def _playback(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        cfg = self.config.playback
        if self.speaker_models:
            model = self.speaker_models[int(rng.integers(len(self.speaker_models)))]
        else:
            weights = np.asarray(cfg.quality_weights, dtype=float)
            quality = cfg.quality[int(rng.choice(len(cfg.quality), p=weights / weights.sum()))]
            model = synth_speaker_kernels(self.sr, rng, quality, cfg.n_orders)
        drive = sample_range(rng, cfg.drive_db)
        return apply_speaker(x, self.sr, model, rng, drive_db=drive)

    def _capture(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        cfg = self.config.capture
        meta: dict = {}

        mic_ir = None
        if self.mic_bank is not None:
            mic_ir, label = self.mic_bank.sample(rng)
            meta["mic_ir"] = label
        x, mic_meta = apply_mic_response(x, self.sr, rng, mic_ir=mic_ir)
        meta.update(mic_meta)

        if cfg.agc:
            target = sample_range(rng, cfg.agc_target_dbfs)
            x, agc_meta = simple_agc(x, self.sr, target, cfg.agc_max_gain_db)
            meta.update(agc_meta)

        x, clip_meta = adc_clip(x, sample_range(rng, cfg.adc_headroom_db))
        meta.update(clip_meta)
        return x, meta

    def _transport(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int, dict]:
        cfg = self.config.transport
        names, weights = self.codec_names, self.codec_weights
        name = names[int(rng.choice(len(names), p=weights))]
        spec = (
            CodecSpec(name, int(sample_range(rng, cfg.bitrate_bps))) if name is not None else None
        )
        drift = sample_range(rng, cfg.clock_drift_ppm)
        return apply_transport(x, self.sr, self.config.output.sr, spec, drift)


def _place_source_and_mic(
    dims: list[float], distance_spec, rng: np.random.Generator
) -> tuple[list[float], list[float], float]:
    """Pick loudspeaker and microphone positions a sampled distance apart.

    Drawing both positions independently would sometimes put them on top of each
    other, which produces a degenerate IR with no meaningful DRR.
    """
    margin = 0.5
    target = sample_range(rng, distance_spec)
    max_span = float(np.linalg.norm([d - 2 * margin for d in dims]))
    target = min(target, max_span * 0.9)

    for _ in range(50):
        src = [float(rng.uniform(margin, d - margin)) for d in dims]
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-12
        mic = [s + target * float(v) for s, v in zip(src, direction, strict=True)]
        if all(margin <= m <= d - margin for m, d in zip(mic, dims, strict=True)):
            return src, mic, target

    centre = [d / 2 for d in dims]
    offset = min(target, min(dims) / 2 - margin)
    return centre, [centre[0] + offset, centre[1], centre[2]], offset


def describe_config(chain: ReplayChain) -> dict:
    """Summarise what the chain will actually do, after resolving optional resources."""
    return {
        "sr_internal": chain.sr,
        "sr_out": chain.config.output.sr,
        "room_mode": chain.config.room.mode,
        "room_irs": len(chain.ir_bank) if chain.ir_bank else 0,
        "mic_irs": len(chain.mic_bank) if chain.mic_bank else 0,
        "noise_files": len(chain.noise_bank.paths) if chain.noise_bank else 0,
        "measured_speakers": len(chain.speaker_models),
        "codecs": chain.codec_names,
    }


__all__ = ["ChainResult", "ReplayChain", "describe_config", "variant_seed"]
