"""
Nonlinear dynamics measures for voice analysis — Zone B.

RPDE (Recurrence Period Density Entropy), DFA (Detrended Fluctuation
Analysis) and PPE (Pitch Period Entropy), implemented directly in numpy.

Why these live here instead of coming from a library:
`nolds` (which ml/audio.py previously used) has **no rpde() function in any
released version** — the old `nolds.rpde(...)` call would have raised
AttributeError, which a bare `except Exception` turned into a constant 0.0
feature rather than an error. nolds 0.6.3 also fails to import at all on
Python 3.10: its bundled datasets.py calls importlib.resources.files() on a
module rather than a package, which broke `import ml.audio` outright.
Implementing the three measures here drops that dependency entirely.

CALIBRATION CAVEAT — read before trusting these against a UCI-trained model:
the UCI Parkinson's Telemonitoring dataset's rpde/dfa/ppe columns were
produced by Little et al.'s original MATLAB implementations. The functions
below follow the same published definitions, but will not be numerically
identical (different radius/window/bin choices shift the absolute values).
A model trained on the UCI columns is therefore not strictly calibrated to
features extracted here. That mismatch is not introduced by this module —
it would have existed with nolds too. Resolving it properly means either
training on features extracted by this code, or calibrating against the
reference implementation.

References:
  Little et al. (2007), "Exploiting nonlinear recurrence and fractal scaling
  properties for voice disorder detection", BioMedical Engineering OnLine.
  Peng et al. (1994), "Mosaic organization of DNA nucleotides", Phys. Rev. E.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dfa", "rpde", "ppe"]


def dfa(x, nvals: np.ndarray | None = None, order: int = 1) -> float:
    """Detrended Fluctuation Analysis scaling exponent (alpha).

    Integrates the mean-removed signal, then measures how the RMS of
    polynomial-detrended residuals grows with window size. The slope of
    log F(n) vs log n is the scaling exponent.

    Reference values: uncorrelated white noise -> ~0.5, pink (1/f) noise
    -> ~1.0, Brownian motion / random walk -> ~1.5.

    Args:
        x: 1-D signal.
        nvals: Window sizes to evaluate. Defaults to ~20 log-spaced sizes
            between 4 and len(x)//4.
        order: Order of the polynomial detrend within each window (1 = linear).

    Returns:
        The scaling exponent alpha.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n_samples = x.size
    if n_samples < 32:
        raise ValueError(f"DFA needs at least 32 samples, got {n_samples}")

    # "Profile": cumulative sum of the mean-removed signal.
    profile = np.cumsum(x - x.mean())

    if nvals is None:
        max_n = max(8, n_samples // 4)
        nvals = np.unique(np.logspace(np.log10(4), np.log10(max_n), 20).astype(int))
    nvals = np.asarray([n for n in nvals if order + 2 <= n <= n_samples // 2], dtype=int)
    if nvals.size < 2:
        raise ValueError("DFA needs at least 2 usable window sizes for the log-log fit")

    fluctuations, used_nvals = [], []
    for n in nvals:
        n_windows = n_samples // n
        if n_windows < 1:
            continue
        segments = profile[: n_windows * n].reshape(n_windows, n)

        # Least-squares polynomial detrend of every window at once.
        t = np.arange(n, dtype=float)
        vander = np.vander(t, order + 1)
        coeffs, *_ = np.linalg.lstsq(vander, segments.T, rcond=None)
        residuals = segments - (vander @ coeffs).T

        f_n = np.sqrt(np.mean(residuals**2))
        if f_n > 0:
            fluctuations.append(f_n)
            used_nvals.append(n)

    if len(fluctuations) < 2:
        raise ValueError("DFA could not compute enough non-zero fluctuations")

    slope = np.polyfit(np.log(used_nvals), np.log(fluctuations), 1)[0]
    return float(slope)


def rpde(
    x,
    emb_dim: int = 4,
    delay: int = 1,
    radius: float | None = None,
    max_period: int = 2000,
    max_points: int = 1500,
) -> float:
    """Recurrence Period Density Entropy, normalised to [0, 1].

    Time-delay embeds the signal, then for each reference point measures the
    *recurrence period*: how long the trajectory takes to return inside a
    ball of `radius` around that point, having first left it. The normalised
    Shannon entropy of the resulting period histogram is the RPDE.

    Periodic signals return at one consistent period -> narrow histogram ->
    RPDE near 0. Aperiodic/noisy signals return at scattered times -> broad
    histogram -> RPDE near 1.

    Args:
        x: 1-D signal.
        emb_dim: Embedding dimension.
        delay: Embedding delay in samples.
        radius: Recurrence ball radius, in units of the variance-normalised
            signal. Defaults to the 5th percentile of sampled pairwise
            distances, which adapts to the attractor's size.
        max_period: Longest recurrence period to look for, in samples. Also
            sets the histogram length used for entropy normalisation.
        max_points: Number of reference points sampled evenly along the
            signal. Caps the cost at O(max_points * max_period).

    Returns:
        RPDE in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    std = x.std()
    if std <= 0:
        raise ValueError("RPDE needs a non-constant signal")
    # Variance-normalise so `radius` is scale-free.
    x = (x - x.mean()) / std

    span = (emb_dim - 1) * delay
    n_vectors = x.size - span
    if n_vectors < 100:
        raise ValueError(f"RPDE needs a longer signal (got {n_vectors} embedded vectors)")

    offsets = np.arange(emb_dim) * delay
    embedded = x[np.arange(n_vectors)[:, None] + offsets]  # (n_vectors, emb_dim)

    if radius is None:
        # Deterministic, evenly-spaced sample of pairs to size the ball.
        sample_idx = np.linspace(0, n_vectors - 1, min(1000, n_vectors)).astype(int)
        sample = embedded[sample_idx]
        dists = np.linalg.norm(sample[: len(sample) // 2] - sample[len(sample) // 2 :][: len(sample) // 2], axis=1)
        positive = dists[dists > 0]
        radius = float(np.percentile(positive, 5)) if positive.size else 0.1

    last_start = max(1, n_vectors - max_period - 1)
    ref_idx = np.unique(np.linspace(0, last_start, min(max_points, last_start)).astype(int))

    periods: list[int] = []
    for i in ref_idx:
        window = embedded[i + 1 : i + 1 + max_period]
        if window.shape[0] < 2:
            continue
        dist = np.linalg.norm(window - embedded[i], axis=1)
        inside = dist <= radius
        if inside.all():
            continue  # never leaves the ball; no period observable here
        exit_at = int(np.argmin(inside))  # first sample outside the ball
        after_exit = inside[exit_at:]
        if not after_exit.any():
            continue  # never comes back within max_period
        periods.append(int(np.argmax(after_exit)) + exit_at + 1)

    if not periods:
        raise ValueError("RPDE found no recurrences — try a larger radius or max_period")

    counts = np.bincount(periods, minlength=max_period + 1)[1:]
    probs = counts / counts.sum()
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    return float(max(0.0, entropy / np.log(probs.size)))


def ppe(
    f0_values,
    n_bins: int = 30,
    lpc_order: int = 2,
    bin_range_semitones: float = 2.0,
) -> float:
    """Pitch Period Entropy, normalised to [0, 1].

    Measures how much *unpredictable* pitch variation a speaker has, on a
    perceptual (log/semitone) scale. Healthy speech carries smooth, largely
    predictable intonation; the residual left after removing that smooth
    component is what this scores.

    Steps: voiced F0 -> semitones relative to a robust baseline -> remove the
    linearly-predictable component -> normalised Shannon entropy of the
    residual, binned on a *fixed* semitone scale.

    The bin edges are deliberately fixed rather than fitted to the data's own
    range: with auto-scaled bins any residual, however tiny, spreads across
    every bin and scores near-maximum entropy, so a perfectly smooth vibrato
    would look as erratic as genuinely unstable pitch.

    Args:
        f0_values: F0 estimates in Hz. Unvoiced frames (<= 0) are dropped.
        n_bins: Histogram bins used for the discrete entropy.
        lpc_order: Order of the linear predictor used for whitening.
        bin_range_semitones: Half-width of the fixed histogram range, in
            semitones. Residuals beyond it are clipped into the end bins.

    Returns:
        PPE in [0, 1].
    """
    f0 = np.asarray(f0_values, dtype=float)
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if f0.size < 20:
        raise ValueError(f"PPE needs at least 20 voiced frames, got {f0.size}")

    # Perceptual scale: semitones above a robust (10th-percentile) baseline.
    baseline = np.percentile(f0, 10)
    if baseline <= 0:
        raise ValueError("PPE baseline pitch must be positive")
    semitones = 12.0 * np.log2(f0 / baseline)

    residual = _lpc_residual(semitones, order=lpc_order)
    if residual.size < 2 or not np.any(np.isfinite(residual)):
        raise ValueError("PPE residual could not be computed")

    clipped = np.clip(residual, -bin_range_semitones, bin_range_semitones)
    counts, _ = np.histogram(
        clipped, bins=n_bins, range=(-bin_range_semitones, bin_range_semitones)
    )
    probs = counts / counts.sum()
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    return float(max(0.0, entropy / np.log(n_bins)))


def _lpc_residual(signal: np.ndarray, order: int) -> np.ndarray:
    """Residual of `signal` after removing its linearly-predictable part.

    Predicts each sample from the previous `order` samples by least squares
    and returns what the predictor could not explain.
    """
    n = signal.size
    if n <= order + 1:
        return signal - signal.mean()
    design = np.column_stack([signal[order - 1 - k : n - 1 - k] for k in range(order)])
    target = signal[order:]
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coeffs
