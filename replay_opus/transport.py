"""Transport stage: sample-rate conversion, clock drift, lossy codecs.

Whatever captures the replay almost never hands you raw PCM at the chain's internal
rate. The codec list mirrors the conditions ASVspoof 5 applies to its evaluation set
(Opus, AMR, MP3, M4A/AAC), which is the closest thing to a standard for this stage.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import av
import numpy as np

from .dsp import align_to, apply_clock_drift, resample_to

# codec name -> (container format, default bitrate, allowed sample rates or None)
CODECS: dict[str, tuple[str, int, tuple[int, ...] | None]] = {
    "opus": ("ogg", 24_000, (48_000, 24_000, 16_000, 12_000, 8_000)),
    "aac": ("adts", 64_000, (48_000, 44_100, 32_000, 24_000, 16_000)),
    "mp3": ("mp3", 64_000, (48_000, 44_100, 32_000, 24_000, 16_000)),
    "pcm_mulaw": ("wav", 64_000, (8_000, 16_000)),
    # Needs an ffmpeg build with libopencore-amrnb; see available_codecs().
    "amr_nb": ("amr", 12_200, (8_000,)),
}

_ENCODER_NAMES = {"opus": "libopus", "mp3": "libmp3lame"}


@lru_cache(maxsize=1)
def available_codecs() -> tuple[str, ...]:
    """Codecs whose encoder this PyAV build can actually open.

    A codec can be registered by name yet unusable, which only surfaces when the
    encoder is opened - so probe rather than trust the registry.
    """
    ok = []
    for name, (container, bitrate, rates) in CODECS.items():
        encoder = _ENCODER_NAMES.get(name, name)
        rate = (rates or (48_000,))[0]
        try:
            with av.open(io.BytesIO(), mode="w", format=container) as out:
                stream = out.add_stream(encoder, rate=rate, layout="mono")
                stream.bit_rate = bitrate
                frame = av.AudioFrame.from_ndarray(
                    np.zeros((1, getattr(stream.codec_context, "frame_size", 0) or 1024), np.int16),
                    format="s16",
                    layout="mono",
                )
                frame.sample_rate = rate
                frame.pts = 0
                stream.encode(frame)
        except Exception:  # noqa: BLE001, S112 - unusable encoder, just exclude it
            continue
        ok.append(name)
    return tuple(ok)


@dataclass
class CodecSpec:
    name: str
    bitrate: int | None = None

    def resolve(self) -> tuple[str, str, int, tuple[int, ...] | None]:
        if self.name not in CODECS:
            raise ValueError(f"unknown codec {self.name!r}; choose from {list(CODECS)}")
        container, default_bitrate, rates = CODECS[self.name]
        encoder = _ENCODER_NAMES.get(self.name, self.name)
        return encoder, container, self.bitrate or default_bitrate, rates


def _nearest_rate(sr: int, allowed: tuple[int, ...] | None) -> int:
    if not allowed:
        return sr
    return min(allowed, key=lambda r: abs(r - sr))


def codec_roundtrip(x: np.ndarray, sr: int, spec: CodecSpec) -> tuple[np.ndarray, dict]:
    """Encode to a lossy codec in memory and decode straight back.

    The signal is resampled to a rate the encoder accepts and back again, so the
    returned array is at the input rate. Length is restored to the input length -
    codecs add priming/padding samples that would otherwise accumulate.
    """
    encoder, container, bitrate, allowed = spec.resolve()
    enc_sr = _nearest_rate(sr, allowed)
    n_in = len(x)

    staged = resample_to(x, sr, enc_sr)
    pcm = (np.clip(staged, -1.0, 1.0) * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with av.open(buf, mode="w", format=container) as out:
        # Without an explicit mono layout the stream defaults to stereo, and ffmpeg's
        # mono->stereo upmix applies a 1/sqrt(2) power-normalising matrix.
        stream = out.add_stream(encoder, rate=enc_sr, layout="mono")
        stream.bit_rate = int(bitrate)
        frame_size = getattr(stream.codec_context, "frame_size", 0) or 1024

        pad = (-len(pcm)) % frame_size
        padded = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)]) if pad else pcm

        for start in range(0, len(padded), frame_size):
            frame = av.AudioFrame.from_ndarray(
                padded[start : start + frame_size].reshape(1, -1), format="s16", layout="mono"
            )
            frame.sample_rate = enc_sr
            frame.pts = start
            frame.time_base = Fraction(1, enc_sr)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)

    payload_bytes = len(buf.getvalue())
    buf.seek(0)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=enc_sr)
    decoded: list[np.ndarray] = []
    with av.open(buf, mode="r", format=container) as inp:
        for frame in inp.decode(audio=0):
            for out_frame in resampler.resample(frame):
                decoded.append(out_frame.to_ndarray().reshape(-1))
        for out_frame in resampler.resample(None):
            decoded.append(out_frame.to_ndarray().reshape(-1))

    if not decoded:
        raise RuntimeError(f"codec {spec.name} produced no audio")

    y = np.concatenate(decoded).astype(np.float64) / 32768.0
    y, lag = align_to(y, staged)
    y = resample_to(y, enc_sr, sr)
    y = y[:n_in] if len(y) >= n_in else np.pad(y, (0, n_in - len(y)))
    return y, {
        "codec": spec.name,
        "codec_bitrate": int(bitrate),
        "codec_sr": enc_sr,
        "codec_bytes": int(payload_bytes),
        "codec_delay_samples": int(lag),
    }


def apply_transport(
    x: np.ndarray,
    sr: int,
    out_sr: int,
    spec: CodecSpec | None,
    clock_drift_ppm: float = 0.0,
) -> tuple[np.ndarray, int, dict]:
    """Clock drift, then codec, then the final rate conversion.

    Drift comes first because it happens in the analogue-to-digital hop, before
    anything is encoded.
    """
    meta: dict = {"clock_drift_ppm": round(float(clock_drift_ppm), 2)}
    y = apply_clock_drift(x, clock_drift_ppm)

    if spec is not None:
        y, codec_meta = codec_roundtrip(y, sr, spec)
        meta.update(codec_meta)
    else:
        meta["codec"] = None

    y = resample_to(y, sr, out_sr)
    meta["out_sr"] = out_sr
    return y, out_sr, meta
