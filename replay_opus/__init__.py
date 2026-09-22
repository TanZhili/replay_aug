"""Batch simulation of loudspeaker playback and microphone re-recording.

Turns clean or TTS-synthesised audio into audio that behaves like it was played in a
room and captured again, by cascading the five stages a real replay goes through:
playback device, room, environmental noise, capture device, transport.

    from replay_opus import Config, ReplayChain

    chain = ReplayChain(Config())
    result = chain.process(audio, sr, key="utt_0001", variant=0)
"""

from .chain import ChainResult, ReplayChain, describe_config
from .config import Config
from .irbank import IRBank, reshape_ir, synth_ir
from .measure import SweepSpec, generate_sweep, measure_chain
from .metrics import describe_ir, estimate_c50, estimate_drr, estimate_t60

__version__ = "0.1.0"

__all__ = [
    "ChainResult",
    "Config",
    "IRBank",
    "ReplayChain",
    "SweepSpec",
    "describe_config",
    "describe_ir",
    "estimate_c50",
    "estimate_drr",
    "estimate_t60",
    "generate_sweep",
    "measure_chain",
    "reshape_ir",
    "synth_ir",
]
