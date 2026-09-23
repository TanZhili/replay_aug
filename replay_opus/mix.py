"""Microphone mix of a user and a playback device in one environment.

``render_mic`` takes four files. Each one may be omitted:

* ``speech`` — the user (A). A mouth, a talking level, a facing direction.
* ``playback`` — audio rendered as an electronic device (B): programme chain,
  loudspeaker, then the room. A and B may be TTS from the same batch, or the
  same file.
* ``noise`` — a noise recording at the microphone. Omitted, the scene uses its
  synthetic ambience, scaled to the scene's A-weighted level either way.
* ``rir`` — one measured room impulse response, shared by every source.
  Omitted, the geometric room is used. A measured IR's own gain is discarded
  and its direct path is scaled to the source distance, as with ``room.measured``.

Both sources start together and overlap; the longer one continues alone. The
return value is what the scene's microphone records, after the shared capture
chain.

``main`` reads one optional path list for each of those four inputs and writes
one mix per line. Lists are aligned by line and shorter lists cycle. The list
format is the same as everywhere else: one path per line, optional text and
speaker on the speech and playback lists.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .config import Config
from .dsp import (
    a_weight,
    fade_edges,
    fit_length,
    resample_to,
    rms,
    sample_range,
    speech_segments,
    to_mono,
)
from .irbank import IR_SUFFIXES, IRBank
from .noise import scale_to_dba
from .pathlist import read_path_list, read_utterances
from .scene import SceneRenderer, derive_seed, spl_dba, stream_rng

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus")
MIN_SEGMENT_S = 0.05
_CACHE_SIZE = 32
_STATE: dict[str, object] = {}


@dataclass
class MicRender:
    audio: np.ndarray
    sr: int
    meta: dict


def render_mic(
    speech: str | Path | None = None,
    playback: str | Path | None = None,
    noise: str | Path | None = None,
    rir: str | Path | None = None,
    *,
    config: Config | None = None,
    scenes: SceneRenderer | None = None,
    seed: int = 0,
    key: str = "mic",
    cache: dict | None = None,
) -> MicRender:
    """Record one environment at its microphone.

    ``speech`` is the user and ``playback`` is rendered as a device. ``noise``
    and ``rir`` replace the synthetic ambience and the geometric room when a
    recording is available. Any argument may be ``None``.
    """
    if scenes is None:
        scenes = SceneRenderer(config or Config())
    config = scenes.config
    cache = {} if cache is None else cache
    sr = scenes.sr
    speech_path = _require_file(speech, "audio") if speech is not None else None
    playback_path = _require_file(playback, "audio") if playback is not None else None
    noise_path = _require_file(noise, "noise") if noise is not None else None
    rir_path = _checked_rir(rir) if rir is not None else None

    scene = scenes.sample_scene(int(seed), key)
    if rir_path is not None:
        _install_rir(scenes, scene, rir_path, cache)
    dry_user = _load_dry(speech_path, sr, cache) if speech_path is not None else None
    dry_device = _load_dry(playback_path, sr, cache) if playback_path is not None else None

    if dry_user is None and dry_device is None:
        seconds = sample_range(stream_rng(int(seed), "compose"), config.scene.duration_s)
        n = max(1, int(seconds * sr))
    else:
        n = max(len(dry) for dry in (dry_user, dry_device) if dry is not None)
        n += int(sr * config.render.tail_ms / 1000.0)

    mix = np.zeros(n)
    sources: dict[str, dict] = {}
    segments: list[dict] = []
    if dry_user is not None:
        _add_source(
            mix,
            segments,
            sources,
            scenes.render_talker(scene, dry_user),
            dry_user,
            sr,
            n,
            "live",
            "live0",
        )
    if dry_device is not None:
        _add_source(
            mix,
            segments,
            sources,
            scenes.render_device(scene, dry_device),
            dry_device,
            sr,
            n,
            "device",
            "device0",
        )

    if noise_path is not None:
        bed = _cached_audio(noise_path, sr, cache, "noise")
        noise_pa = _slice_noise(bed, n, sr, scene.noise_dba, stream_rng(scene.seed, "noise", 1))
        noise_label = noise_path.name
    elif config.noise.enabled:
        noise_pa, noise_label = scenes.ambient(scene, n)
    else:
        noise_pa, noise_label = np.zeros(n), "none"

    audio, capture_meta = scenes.capture(scene, mix + noise_pa)
    duration = len(audio) / config.output.sr
    for segment in segments:
        segment["end"] = min(segment["end"], round(duration, 3))
    segments = [s for s in segments if s["end"] - s["start"] >= MIN_SEGMENT_S]
    meta = {
        "key": key,
        "seed": int(seed),
        "sr": config.output.sr,
        "duration_s": round(duration, 3),
        "inputs": {
            "speech": None if speech_path is None else str(speech_path),
            "playback": None if playback_path is None else str(playback_path),
            "noise": None if noise_path is None else str(noise_path),
            "rir": None if rir_path is None else str(rir_path),
        },
        "segments": segments,
        "sources": sources,
        "scene": {**scene.describe(), "noise_source": noise_label},
        "capture_result": capture_meta,
    }
    return MicRender(audio, config.output.sr, meta)


def main(
    speech_list: str | Path | None = None,
    playback_list: str | Path | None = None,
    noise_list: str | Path | None = None,
    rir_list: str | Path | None = None,
    output: str | Path = "out/mix",
    *,
    config: Config | None = None,
    count: int | None = None,
    workers: int = 1,
) -> int:
    """Render one mix per line of the four lists. Every list may be omitted.

    Line ``i`` takes entry ``i % length`` from each list that was given, so a
    short noise or RIR list repeats and the same list may be passed as both the
    user speech and the playback audio. ``count`` overrides the number of mixes;
    without it, the longest list sets the count.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if count is not None and count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    config = config or Config()
    config.validate()

    speeches = read_utterances(speech_list, suffixes=AUDIO_SUFFIXES) if speech_list else []
    playbacks = read_utterances(playback_list, suffixes=AUDIO_SUFFIXES) if playback_list else []
    noises = (
        read_path_list(noise_list, kind="noise", suffixes=AUDIO_SUFFIXES) if noise_list else []
    )
    rirs = (
        read_path_list(rir_list, kind="impulse response", suffixes=IR_SUFFIXES) if rir_list else []
    )
    n_mixes = count if count is not None else max(
        (len(speeches), len(playbacks), len(noises), len(rirs)), default=0
    )
    if n_mixes == 0:
        raise ValueError(
            "pass at least one of speech_list, playback_list, noise_list, rir_list, or set count"
        )

    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    _save_config(config, out_root)
    suffix = config.output.suffix
    jobs = []
    for index in range(n_mixes):
        speech = _at(speeches, index)
        played = _at(playbacks, index)
        noise = _at(noises, index)
        rir = _at(rirs, index)
        name = f"mix_{index:06d}"
        rel = f"{name}{suffix}"
        jobs.append(
            (
                str(speech.path) if speech else None,
                str(played.path) if played else None,
                str(noise) if noise else None,
                str(rir) if rir else None,
                str(out_root / rel),
                rel,
                name,
                derive_seed(config.seed, "mix", index),
                {
                    "text": None if speech is None else speech.text,
                    "speaker": None if speech is None else speech.speaker,
                    "playback_text": None if played is None else played.text,
                    "playback_speaker": None if played is None else played.speaker,
                },
            )
        )
    print(
        f"{n_mixes} mixes  speech {len(speeches)}  playback {len(playbacks)}  "
        f"noise {len(noises)}  rir {len(rirs)}"
    )
    manifest = out_root / "manifest.jsonl"
    written, failed = _run_jobs(jobs, config, workers, manifest)
    print(f"wrote {written} files, {failed} failed\nmanifest: {manifest}")
    return 1 if failed and not written else 0


def _add_source(mix, segments, sources, source, dry, sr, n, role, name) -> None:
    signal = fit_length(source.signal, n)
    mix[: len(signal)] += signal
    duration = n / sr
    for raw_start, raw_end in speech_segments(dry, sr):
        start = max(0.0, raw_start + source.delay_s)
        end = min(duration, raw_end + source.delay_s)
        if end - start < MIN_SEGMENT_S:
            continue
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "role": role,
                "source": name,
            }
        )
    level = spl_dba(signal, sr)
    sources[name] = {
        **source.meta,
        "speech_dba_at_mic": None if not np.any(signal) else round(level, 2),
    }


def _checked_rir(rir: str | Path) -> Path:
    path = _require_file(rir, "room impulse response")
    if path.suffix.lower() not in IR_SUFFIXES:
        expected = " ".join(IR_SUFFIXES)
        raise ValueError(f"{path} is not a room impulse response (expected {expected})")
    return path


def _install_rir(scenes: SceneRenderer, scene, path: Path, cache: dict) -> None:
    banks: OrderedDict = cache.setdefault("rir", OrderedDict())
    bank = banks.get(path)
    if bank is None:
        if len(banks) >= _CACHE_SIZE:
            banks.popitem(last=False)
        bank = IRBank(root=path.parent, sr=scenes.sr, paths=[path], labels=[path.name])
        banks[path] = bank
    else:
        banks.move_to_end(path)
    scene.measured = {"bank": bank, "indices": [0], "room": path.stem}
    scene.used_irs = []


def _load_dry(path: str | Path, sr: int, cache: dict) -> np.ndarray:
    found = _require_file(path, "audio")
    audio = _cached_audio(found, sr, cache, "dry")
    return fade_edges(audio.copy(), sr, 5.0)


def _cached_audio(path: Path, sr: int, cache: dict, bucket: str) -> np.ndarray:
    store: OrderedDict = cache.setdefault(bucket, OrderedDict())
    key = (path, sr)
    if key not in store:
        if len(store) >= _CACHE_SIZE:
            store.popitem(last=False)
        data, sr_file = sf.read(str(path), always_2d=False)
        if np.asarray(data).size == 0:
            raise ValueError(f"empty audio file: {path}")
        store[key] = resample_to(to_mono(data), sr_file, sr)
    store.move_to_end(key)
    return store[key]


def _slice_noise(
    audio: np.ndarray, n: int, sr: int, level_dba: float, rng: np.random.Generator
) -> np.ndarray:
    if len(audio) < n:
        audio = np.tile(audio, int(np.ceil(n / max(len(audio), 1))))
    start = int(rng.integers(0, max(1, len(audio) - n + 1)))
    clip = audio[start : start + n].copy()
    if rms(a_weight(clip, sr)) < 1e-8:
        return np.zeros(n)
    return scale_to_dba(clip, sr, level_dba)


def _require_file(path: str | Path, kind: str) -> Path:
    found = Path(path)
    if found.is_dir():
        raise ValueError(f"{found} is a directory; {kind} must be a file")
    if not found.is_file():
        raise FileNotFoundError(f"{kind} not found: {found}")
    return found


def _at(items, index: int):
    if not items:
        return None
    return items[index % len(items)]


def _save_config(config: Config, out_root: Path) -> None:
    text = yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False)
    (out_root / "config.yaml").write_text(text, encoding="utf-8")


def _init_worker(config_dict: dict) -> None:
    _STATE["scenes"] = SceneRenderer(Config.from_dict(config_dict))
    _STATE["cache"] = {}


def _mix_job(job: tuple) -> dict:
    speech, playback, noise, rir, target, rel, key, seed, extra = job
    try:
        rendered = render_mic(
            speech,
            playback,
            noise,
            rir,
            scenes=_STATE["scenes"],
            seed=seed,
            key=key,
            cache=_STATE["cache"],
        )
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, rendered.audio, rendered.sr, subtype=_STATE["scenes"].config.output.subtype)
        return {"audio": rel, **extra, **rendered.meta}
    except Exception as exc:  # noqa: BLE001 - one bad file should not stop the batch
        return {"audio": rel, "job": key, "error": repr(exc), **extra}


def _run_jobs(jobs, config: Config, workers: int, manifest_path: Path) -> tuple[int, int]:
    written = failed = 0
    total = len(jobs)

    def record_all(records: list[dict], manifest) -> None:
        nonlocal written, failed
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            if "error" in record:
                failed += 1
                print(f"  FAILED {record.get('job')}: {record['error']}", file=sys.stderr)
            else:
                written += 1

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        if workers == 1:
            _init_worker(config.to_dict())
            for done, job in enumerate(jobs, start=1):
                record_all([_mix_job(job)], manifest)
                if done % 10 == 0 or done == total:
                    print(f"  {done}/{total}", end="\r", flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(config.to_dict(),),
            ) as pool:
                futures = [pool.submit(_mix_job, job) for job in jobs]
                for done, future in enumerate(as_completed(futures), start=1):
                    record_all([future.result()], manifest)
                    if done % 10 == 0 or done == total:
                        print(f"  {done}/{total}", end="\r", flush=True)
    print()
    return written, failed


__all__ = ["MicRender", "main", "render_mic"]
