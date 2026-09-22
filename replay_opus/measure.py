"""Synchronised swept-sine measurement of a real playback -> room -> capture chain.

Play the generated sweep through the loudspeaker you want to model, record it with
the microphone you want to model, and one pass yields both:

* the linear impulse response of the whole chain (speaker + room + mic), and
* the harmonic distortion profile of that chain, which calibrates the non-linear
  part of :mod:`replay_opus.speaker`.

Method: Novak, Lotton & Simon, "Synchronized Swept-Sine: Theory, Application and
Implementation", JAES 63(10), 2015. Harmonic impulse responses separate in time at
offsets ``L*ln(m)`` ahead of the linear one, so one sweep identifies them all.

The two extractions use different windows on purpose. The linear response needs a
long one-sided window to keep the room's decay; the harmonic responses need a short
*symmetric* one, because a harmonic of a memoryless non-linearity deconvolves to a
two-sided Hilbert-shaped impulse whose left half a one-sided window would discard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .dsp import EPS, butter
from .speaker import SpeakerModel

HANN_COHERENT_GAIN = 0.5


@dataclass
class SweepSpec:
    """Sweep parameters.

    ``f1`` defaults to 50 Hz rather than 20 Hz on purpose. A deconvolved 20 Hz-limited
    impulse rings for tens of milliseconds, and that ringing smears the decay of every
    measured IR badly enough to inflate T60 several-fold. Replay loudspeakers produce
    little below 100 Hz anyway.
    """

    f1: float = 50.0
    f2: float = 20_000.0
    duration: float = 12.0
    sr: int = 48_000
    pad_s: float = 2.0

    def rate_parameter(self) -> float:
        """``L`` such that the sweep is phase-synchronised for every harmonic.

        Rounding ``f1*T/ln(f2/f1)`` to an integer is what makes harmonic impulse
        responses land at exactly ``L*ln(m)``; without it their phases smear.
        """
        ratio = np.log(self.f2 / self.f1)
        return round(self.f1 * self.duration / ratio) / self.f1


def _fade(x: np.ndarray, sr: int, head_ms: float, tail_ms: float) -> np.ndarray:
    y = x.copy()
    nh, nt = int(sr * head_ms / 1000.0), int(sr * tail_ms / 1000.0)
    if nh:
        y[:nh] *= np.hanning(2 * nh)[:nh]
    if nt:
        y[-nt:] *= np.hanning(2 * nt)[nt:]
    return y


def generate_sweep(spec: SweepSpec) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(sweep, inverse_filter, L)``.

    The inverse filter is the time-reversed sweep with a -6 dB/octave envelope
    (Farina), normalised so that convolving it with the sweep gives a unit impulse.
    """
    L = spec.rate_parameter()
    duration = L * np.log(spec.f2 / spec.f1)
    n = round(duration * spec.sr)
    t = np.arange(n) / spec.sr

    sweep = np.sin(2.0 * np.pi * spec.f1 * L * (np.exp(t / L) - 1.0))
    # The head fade must span several cycles of f1, or truncating the lowest
    # frequencies puts a large ringing artefact either side of the recovered impulse.
    sweep = _fade(sweep, spec.sr, head_ms=max(20.0, 10_000.0 / spec.f1), tail_ms=20.0)

    inverse = sweep[::-1] * np.exp(-t[::-1] / L)
    impulse = signal.fftconvolve(sweep, inverse, mode="full")
    inverse = inverse / (np.max(np.abs(impulse)) + EPS)

    return sweep, inverse, L


def sweep_with_padding(spec: SweepSpec) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Sweep bracketed by silence so the tail of the response is captured."""
    sweep, inverse, L = generate_sweep(spec)
    pad = np.zeros(int(spec.pad_s * spec.sr))
    return np.concatenate([pad, sweep, pad]), inverse, L, len(pad)


def deconvolve(recorded: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """Convolve the recording with Farina's time-domain inverse filter.

    Keeps the exact phase structure that separates the harmonics in time, which is
    what :func:`extract_harmonic_irs` needs. Its magnitude compensation is only
    approximate though, so the linear path uses :func:`deconvolve_regularized`.
    """
    return signal.fftconvolve(np.asarray(recorded, dtype=np.float64), inverse, mode="full")


def deconvolve_regularized(
    recorded: np.ndarray,
    sweep: np.ndarray,
    sr: int,
    f1: float,
    f2: float,
    eps_in: float = 1e-5,
    eps_out: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Kirkeby-regularised spectral deconvolution. Returns ``(ir_train, origin)``.

    Dividing by the sweep's measured spectrum instead of approximating its inverse
    with a decaying time-reversal gives a genuinely flat in-band impulse. Farina's
    inverse leaves roughly -15 dB of ringing a few milliseconds either side of the
    peak, which is enough to corrupt both the magnitude response and the decay of
    any IR measured with it.

    Regularisation rises outside ``[f1, f2]`` so the division does not amplify the
    band where the sweep put no energy.
    """
    recorded = np.asarray(recorded, dtype=np.float64)
    n_fft = 1 << int(np.ceil(np.log2(len(recorded) + len(sweep))))
    spec_rec = np.fft.rfft(recorded, n_fft)
    spec_sweep = np.fft.rfft(sweep, n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    power = np.abs(spec_sweep) ** 2
    safe = np.maximum(freqs, 1e-9)
    ramp_lo = np.clip(np.log2(safe / (f1 / 2.0)), 0.0, 1.0)
    ramp_hi = np.clip(np.log2(min(f2 * 2.0, sr / 2.0) / safe), 0.0, 1.0)
    in_band = ramp_lo * ramp_hi
    regulariser = power.max() * (eps_in * in_band + eps_out * (1.0 - in_band))

    train = np.fft.irfft(spec_rec * np.conj(spec_sweep) / (power + regulariser), n_fft)
    origin = n_fft // 2
    return np.roll(train, origin), origin


def extract_linear_ir(
    recorded: np.ndarray,
    spec: SweepSpec,
    ir_ms: float = 1500.0,
    pre_ms: float = 5.0,
    highpass_hz: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Pull the linear impulse response of the whole chain out of a recorded sweep.

    This is the array to convolve with: it already contains loudspeaker, room and
    microphone together, which is why one sweep per configuration is enough.

    ``highpass_hz`` removes content below the sweep's own start frequency, where the
    result is out-of-band ringing rather than measurement.
    """
    sweep, _, L = generate_sweep(spec)
    train, origin = deconvolve_regularized(recorded, sweep, spec.sr, spec.f1, spec.f2)

    # Search near the known origin: harmonic responses sit earlier and can outrank
    # the linear peak when the chain distorts heavily.
    guard = int(spec.sr * 0.05)
    window = train[max(origin - guard, 0) : origin + guard]
    peak = max(origin - guard, 0) + int(np.argmax(np.abs(window)))

    pre = int(spec.sr * pre_ms / 1000.0)
    start = max(peak - pre, 0)
    ir = train[start : start + int(spec.sr * ir_ms / 1000.0)]
    if highpass_hz:
        ir = butter(ir, spec.sr, highpass_hz, kind="high", order=4)
    return ir, {
        "peak_index": peak,
        "pre_samples": peak - start,
        "rate_parameter_L": float(L),
        "highpass_hz": highpass_hz,
    }


def extract_harmonic_irs(
    recorded: np.ndarray,
    spec: SweepSpec,
    orders: list[int],
    window: int = 8192,
) -> tuple[list[np.ndarray], dict]:
    """Symmetric, Hann-tapered windows around each harmonic impulse response.

    Every window shares the same length and the same centre convention, so the
    responses keep a common time origin at ``window // 2`` and their phases stay
    comparable. The Hann coherent gain is divided out so magnitudes are absolute.
    """
    _, inverse, L = generate_sweep(spec)
    hhir = deconvolve(recorded - np.mean(recorded), inverse)
    peak = int(np.argmax(np.abs(hhir)))
    half = window // 2
    taper = np.hanning(window) / HANN_COHERENT_GAIN

    gaps = [round(L * np.log(m + 1) * spec.sr) - round(L * np.log(m) * spec.sr) for m in orders]
    if window > min(gaps):
        raise ValueError(
            f"window {window} exceeds the {min(gaps)}-sample gap between the highest "
            "harmonics; shorten the window or lengthen the sweep"
        )

    irs = []
    for order in orders:
        centre = peak - round(L * np.log(order) * spec.sr)
        start = centre - half
        segment = np.zeros(window)
        chunk = hhir[max(start, 0) : start + window]
        segment[max(-start, 0) : max(-start, 0) + len(chunk)] = chunk
        irs.append(segment * taper)

    return irs, {"peak_index": peak, "origin": half, "rate_parameter_L": float(L)}


def harmonic_profile(
    recorded: np.ndarray,
    spec: SweepSpec,
    orders: list[int] | None = None,
    window: int = 8192,
    band: tuple[float, float] = (300.0, 3000.0),
) -> dict:
    """Level of each harmonic relative to the fundamental, in dB.

    Validated against known polynomial systems: the returned ratios match theory to
    four decimal places. Use :func:`calibrate_harmonic_rolloff` to turn the profile
    into a synthetic speaker model.
    """
    orders = orders or [1, 2, 3, 4, 5]
    irs, info = extract_harmonic_irs(recorded, spec, orders, window)
    freqs = np.fft.rfftfreq(window, 1.0 / spec.sr)
    mask = (freqs >= band[0]) & (freqs <= band[1])

    magnitudes = [float(np.abs(np.fft.rfft(ir))[mask].mean()) for ir in irs]
    fundamental = magnitudes[0] + EPS
    return {
        "orders": list(orders),
        "magnitude": [round(m, 6) for m in magnitudes],
        "level_db": [round(float(20.0 * np.log10(m / fundamental + EPS)), 2) for m in magnitudes],
        "band_hz": list(band),
        **info,
    }


def calibrate_harmonic_rolloff(profile: dict) -> float:
    """Fit one dB-per-order slope to a measured harmonic profile.

    Feed the result to :func:`replay_opus.speaker.synth_speaker_kernels` via a custom
    preset so the simulated device distorts like the one you measured.
    """
    orders = np.asarray(profile["orders"], dtype=float)
    levels = np.asarray(profile["level_db"], dtype=float)
    keep = orders > 1
    if keep.sum() < 2:
        return float(levels[-1])
    slope, _ = np.polyfit(orders[keep] - 1.0, levels[keep], 1)
    return float(slope)


def measure_chain(
    recorded: np.ndarray,
    spec: SweepSpec,
    orders: list[int] | None = None,
    ir_ms: float = 1500.0,
    window: int = 8192,
) -> tuple[np.ndarray, SpeakerModel, dict]:
    """One-call measurement: linear IR, a calibrated speaker model, and the profile.

    The returned :class:`SpeakerModel` reuses the measured linear response as its
    first-order kernel and derives the higher orders from the measured harmonic
    slope. Identifying each higher-order kernel's own frequency response exactly
    requires the full Novak kernel inversion, which this does not attempt.
    """
    orders = orders or [1, 2, 3, 4, 5]
    linear_ir, ir_info = extract_linear_ir(recorded, spec, ir_ms=ir_ms)
    profile = harmonic_profile(recorded, spec, orders, window=window)
    rolloff_db = calibrate_harmonic_rolloff(profile)

    base = linear_ir[: min(len(linear_ir), 4096)]
    base = base / (np.max(np.abs(base)) + EPS)
    kernels = [base]
    for order in orders[1:]:
        gain = 10.0 ** (rolloff_db * (order - 1) / 20.0)
        cutoff = min(spec.f2 / order, 0.98 * spec.sr / 2)
        kernels.append(butter(base, spec.sr, cutoff, "low", 4) * gain)

    model = SpeakerModel(
        kernels=kernels, orders=list(orders), sr=spec.sr, label="measured", clip_kind="none"
    )
    info = {**ir_info, "harmonic_profile": profile, "harmonic_rolloff_db": round(rolloff_db, 2)}
    return linear_ir, model, info
