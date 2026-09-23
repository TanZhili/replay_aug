"""Lossy codec round trips through the ffmpeg bundled with PyAV.

Speech that comes out of a TV or a phone has been through a delivery codec first:
AAC / AC-3 / MP2 for broadcast, AAC / Opus / MP3 for streaming, AMR / G.711 / G.722 /
Opus for calls. The same machinery can also model an uplink codec on the capture
side, which then applies to every class alike.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import av
import numpy as np

from .dsp import align_to, choose, resample_to


@dataclass(frozen=True)
class CodecInfo:
    container: str
    encoder: str
    default_bitrate: int
    rates: tuple[int, ...]
    bitrates: tuple[int, ...] | None = None  # the only bitrates the encoder accepts


_AC3_RATES = (
    96_000,
    112_000,
    128_000,
    160_000,
    192_000,
    224_000,
    256_000,
    320_000,
    384_000,
    448_000,
    512_000,
    576_000,
    640_000,
)
_MP2_RATES = (
    32_000,
    48_000,
    56_000,
    64_000,
    80_000,
    96_000,
    112_000,
    128_000,
    160_000,
    192_000,
    224_000,
    256_000,
    320_000,
    384_000,
)
_AMR_NB_RATES = (4_750, 5_150, 5_900, 6_700, 7_400, 7_950, 10_200, 12_200)

CODECS: dict[str, CodecInfo] = {
    "opus": CodecInfo("ogg", "libopus", 32_000, (48_000, 24_000, 16_000, 12_000, 8_000)),
    "aac": CodecInfo("adts", "aac", 128_000, (48_000, 44_100, 32_000, 24_000, 22_050, 16_000)),
    "mp3": CodecInfo(
        "mp3", "libmp3lame", 128_000, (48_000, 44_100, 32_000, 24_000, 22_050, 16_000)
    ),
    "mp2": CodecInfo("mp2", "mp2", 192_000, (48_000, 44_100, 32_000), _MP2_RATES),
    "ac3": CodecInfo("ac3", "ac3", 192_000, (48_000, 44_100, 32_000), _AC3_RATES),
    "eac3": CodecInfo("eac3", "eac3", 96_000, (48_000, 44_100, 32_000)),
    "g722": CodecInfo("wav", "g722", 64_000, (16_000,)),
    "pcm_mulaw": CodecInfo("wav", "pcm_mulaw", 64_000, (8_000,)),
    "pcm_alaw": CodecInfo("wav", "pcm_alaw", 64_000, (8_000,)),
    "amr_nb": CodecInfo("amr", "libopencore_amrnb", 12_200, (8_000,), _AMR_NB_RATES),
}


@lru_cache(maxsize=1)
def available_codecs() -> tuple[str, ...]:
    """Codecs whose encoder this PyAV build can actually open.

    A codec can be registered by name yet unusable, which only surfaces when the
    encoder is opened - so probe rather than trust the registry.
    """
    ok = []
    for name, info in CODECS.items():
        rate = info.rates[0]
        try:
            with av.open(io.BytesIO(), mode="w", format=info.container) as out:
                stream = out.add_stream(info.encoder, rate=rate, layout="mono")
                stream.bit_rate = info.default_bitrate
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


def pick_codec(rng, weights: dict[str, float], available) -> str | None:
    """Draw a codec from ``weights``. ``none``, or no usable codec, means no codec."""
    usable = {k: w for k, w in weights.items() if k == "none" or k in available}
    if not usable:
        return None
    name = choose(rng, usable)
    return None if name == "none" else name


@dataclass
class CodecSpec:
    name: str
    bitrate: int | None = None
    rate: int | None = None  # preferred encoder sample rate

    def info(self) -> CodecInfo:
        if self.name not in CODECS:
            raise ValueError(f"unknown codec {self.name!r}; choose from {list(CODECS)}")
        return CODECS[self.name]

    def resolved_bitrate(self) -> int:
        info = self.info()
        wanted = int(self.bitrate or info.default_bitrate)
        if info.bitrates:
            return min(info.bitrates, key=lambda b: abs(b - wanted))
        return wanted


def encoder_rate(sr: int, allowed: tuple[int, ...], preferred: int | None = None) -> int:
    """The preferred rate if the encoder takes it, else the lowest one at or above ``sr``."""
    if preferred in allowed:
        return int(preferred)
    above = [r for r in allowed if r >= sr]
    return min(above) if above else max(allowed)


def codec_roundtrip(x: np.ndarray, sr: int, spec: CodecSpec) -> tuple[np.ndarray, dict]:
    """Encode to a lossy codec in memory and decode straight back.

    The signal is resampled to a rate the encoder accepts and back again, so the
    returned array is at the input rate. Length is restored to the input length and
    the codec's priming delay removed - codecs add samples that would otherwise
    become a random latency in every file.
    """
    info = spec.info()
    bitrate = spec.resolved_bitrate()
    enc_sr = encoder_rate(sr, info.rates, spec.rate)
    n_in = len(x)

    staged = resample_to(np.asarray(x, dtype=np.float64), sr, enc_sr)
    pcm = (np.clip(staged, -1.0, 1.0) * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with av.open(buf, mode="w", format=info.container) as out:
        # Without an explicit mono layout the stream defaults to stereo, and ffmpeg's
        # mono->stereo upmix applies a 1/sqrt(2) power-normalising matrix.
        stream = out.add_stream(info.encoder, rate=enc_sr, layout="mono")
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
    with av.open(buf, mode="r", format=info.container) as inp:
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
