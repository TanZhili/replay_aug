"""Live talkers: a person speaking in the room, rendered from dry TTS.

The dry signal is taken as the talker's on-axis free-field pressure at 1 m. What is
modelled is what differs from a loudspeaker playing the same words: a human-scale
speech level that rises in noise (the Lombard effect) with the flatter spectrum of
raised voice, and the radiation pattern of a mouth in a head - near-omni at low
frequency, increasingly forward-beamed above 1 kHz (Chu & Warnock, NRC IRC-RR-104,
2002). No band limit, no codec, no loudspeaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .acoustics import Directivity
from .dsp import P_REF, apply_sos, sample_range, shelf_sos, speech_level_db
from .environment import Placement

# Cardioid-family fit to average talker directivity at 125 Hz ... 8 kHz.
MOUTH_DIRECTIVITY = np.array([0.97, 0.93, 0.86, 0.76, 0.66, 0.58, 0.51])
NORMAL_EFFORT_DBA = 60.0  # "normal" vocal effort at 1 m (ANSI S3.5 / ISO 9921)


@dataclass
class Talker:
    placement: Placement
    level_dba: float  # active speech level at 1 m on-axis, Lombard rise included
    lombard_db: float
    tilt_db: float
    directivity: Directivity
    meta: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "talker_level_dba": round(self.level_dba, 2),
            "lombard_db": round(self.lombard_db, 2),
            "effort_tilt_db": round(self.tilt_db, 2),
            "pose": self.placement.poses[0].describe(),
            **self.meta,
        }


def sample_talker(placement: Placement, cfg, noise_dba: float, rng: np.random.Generator) -> Talker:
    """Draw a talker: base vocal effort, Lombard rise for this noise, spectral tilt."""
    base = sample_range(rng, cfg.level_dba)
    rise = sample_range(rng, cfg.lombard_slope) * max(0.0, noise_dba - cfg.lombard_onset_dba)
    rise = min(rise, cfg.lombard_max_db)
    level = base + rise
    tilt = sample_range(rng, cfg.effort_tilt_db_per_db) * (level - NORMAL_EFFORT_DBA)
    pose = placement.poses[0]
    return Talker(
        placement=placement,
        level_dba=float(level),
        lombard_db=float(rise),
        tilt_db=float(tilt),
        directivity=Directivity(pose.axis, MOUTH_DIRECTIVITY),
        meta=dict(placement.meta),
    )


def talker_emission(dry: np.ndarray, sr: int, talker: Talker) -> np.ndarray:
    """On-axis pressure at 1 m, in pascals.

    Raised voice puts relatively more energy above ~1 kHz; soft voice less. The tilt
    goes on before the level is set so the requested level holds afterwards.
    """
    x = apply_sos(np.asarray(dry, dtype=np.float64), shelf_sos(sr, 1200.0, talker.tilt_db, "high"))
    want = 20.0 * np.log10(P_REF) + talker.level_dba
    return x * 10.0 ** ((want - speech_level_db(x, sr)) / 20.0)


__all__ = ["MOUTH_DIRECTIVITY", "Talker", "sample_talker", "talker_emission"]
