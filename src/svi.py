"""Raw SVI fits of implied variance for each expiry, with butterfly and calendar arbitrage checks."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

MIN_POINTS = 8               # fewest options an expiry needs to be fitted
BUTTERFLY_TOLERANCE = 1e-6   # density below minus this counts as butterfly arbitrage
CALENDAR_TOLERANCE = 1e-5    # variance falling by more than this with time counts as calendar arbitrage
RANGE_PAD = 0.01             # how far past the quoted strikes (in log-moneyness) a fit is trusted
GOOD_ENOUGH = 0.006          # a fit this close (0.6 vol points) stops the search for a better start


@dataclass(frozen=True)
class SviSlice:
    """The SVI fit of one expiry. ``params`` are (a, b, rho, m, sigma); k is ln(strike / forward)."""
    days: int
    maturity: float
    forward: float
    params: Tuple[float, float, float, float, float]
    n_points: int
    rmse: float          # fit error in volatility (decimal, so 0.004 is 0.4 vol points)
    k_min: float
    k_max: float
    min_density: float   # lowest value of the butterfly density over the quoted strikes

    @property
    def butterfly_ok(self) -> bool:
        return self.min_density >= -BUTTERFLY_TOLERANCE


def total_variance(k: np.ndarray, params: Sequence[float]) -> np.ndarray:
    """SVI total implied variance w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))."""
    a, b, rho, m, sigma = params
    x = np.asarray(k, dtype=float) - m
    return a + b * (rho * x + np.sqrt(x ** 2 + sigma ** 2))


def butterfly_density(k: np.ndarray, params: Sequence[float]) -> np.ndarray:
    """Gatheral's g(k). Option prices are free of butterfly arbitrage where g(k) >= 0."""
    a, b, rho, m, sigma = params
    k = np.asarray(k, dtype=float)
    x = k - m
    root = np.sqrt(x ** 2 + sigma ** 2)
    w = a + b * (rho * x + root)
    slope = b * (rho + x / root)
    curvature = b * sigma ** 2 / root ** 3
    with np.errstate(divide='ignore', invalid='ignore'):
        g = (1 - k * slope / (2 * w)) ** 2 - slope ** 2 / 4 * (1 / w + 0.25) + curvature / 2
    return np.where(w > 1e-12, g, -1.0)


def fit_slice(k: np.ndarray, iv: np.ndarray, maturity: float, days: int = 0,
              forward: float = 1.0) -> Optional[SviSlice]:
    """
    Fit one expiry's implied volatilities with an SVI curve.

    The error is measured in volatility, and a penalty pushes the fit away from curves that allow
    butterfly arbitrage. Returns None if there are too few points. Repeated fits of the same data are cached.
    """
    k = np.ascontiguousarray(k, dtype=float)
    iv = np.ascontiguousarray(iv, dtype=float)
    if len(k) < MIN_POINTS or maturity <= 0:
        return None
    return _fit_cached(k.tobytes(), iv.tobytes(), float(maturity), int(days), float(forward))


def _model_iv(params: Sequence[float], k: np.ndarray, maturity: float) -> np.ndarray:
    """The implied volatility an SVI curve gives at log-moneyness k."""
    return np.sqrt(np.maximum(total_variance(k, params), 1e-12) / maturity)


def _slice_residuals(params: np.ndarray, k: np.ndarray, iv: np.ndarray, maturity: float,
                     grid: np.ndarray) -> np.ndarray:
    """Volatility errors at the quotes, plus penalties for butterfly arbitrage and negative variance on a grid."""
    penalty = 0.1 * np.minimum(butterfly_density(grid, params), 0.0) \
              + np.minimum(total_variance(grid, params), 0.0)
    return np.concatenate([_model_iv(params, k, maturity) - iv, penalty])


@lru_cache(maxsize=512)
def _fit_cached(k_bytes: bytes, iv_bytes: bytes, maturity: float, days: int, forward: float) -> SviSlice:
    k = np.frombuffer(k_bytes, dtype=float)
    iv = np.frombuffer(iv_bytes, dtype=float)

    total = iv ** 2 * maturity
    grid = np.linspace(k.min() - 0.05, k.max() + 0.05, 60)

    lower = np.array([-1.0, 1e-6, -0.999, k.min() - 0.3, 1e-3])
    upper = np.array([total.max(), 5.0, 0.999, k.max() + 0.3, 2.0])

    # Try the usual start first and only try others if the fit is poor
    best = None
    for rho0, m0, sigma0 in ((-0.5, 0.0, 0.15), (-0.8, 0.05, 0.05), (-0.3, 0.0, 0.3), (0.0, -0.05, 0.2)):
        start = np.clip([0.5 * total.min(), 0.1, rho0, m0, sigma0], lower, upper)
        solution = least_squares(_slice_residuals, start, bounds=(lower, upper), method='trf', max_nfev=100,
                                 args=(k, iv, maturity, grid))
        if best is None or solution.cost < best.cost:
            best = solution
        if np.sqrt(np.mean((_model_iv(best.x, k, maturity) - iv) ** 2)) < GOOD_ENOUGH:
            break

    params = tuple(float(value) for value in best.x)
    quoted = np.linspace(k.min(), k.max(), 60)
    return SviSlice(
        days=int(days), maturity=float(maturity), forward=float(forward), params=params, n_points=len(k),
        rmse=float(np.sqrt(np.mean((_model_iv(params, k, maturity) - iv) ** 2))),
        k_min=float(k.min()), k_max=float(k.max()),
        min_density=float(np.min(butterfly_density(quoted, params)))
    )


def calendar_check(slices: List[SviSlice]) -> Dict[str, object]:
    """
    Check that total variance never falls from one expiry to the next at the same log-moneyness.

    Returns the number of consecutive expiry pairs that violate this, the worst fall, and the pairs (in days).
    """
    pairs = []
    worst = 0.0
    ordered = sorted(slices, key=lambda s: s.maturity)
    for earlier, later in zip(ordered[:-1], ordered[1:]):
        low, high = max(earlier.k_min, later.k_min), min(earlier.k_max, later.k_max)
        if high <= low:
            continue
        k = np.linspace(low, high, 40)
        fall = float(np.min(total_variance(k, later.params) - total_variance(k, earlier.params)))
        if fall < -CALENDAR_TOLERANCE:
            pairs.append((earlier.days, later.days))
            worst = min(worst, fall)
    return {'violations': len(pairs), 'worst': worst, 'pairs': pairs}


def _blend_row(ordered: List[SviSlice], times: np.ndarray, maturity: float, k: np.ndarray) -> np.ndarray:
    """Implied volatility at one maturity for log-moneyness values k, NaN where the fits do not cover them."""
    j = min(int(np.searchsorted(times, maturity, side='right')) - 1, len(ordered) - 2)
    earlier, later = ordered[j], ordered[j + 1]
    weight = float(np.clip((maturity - earlier.maturity) / (later.maturity - earlier.maturity), 0.0, 1.0))
    variance = (1 - weight) * total_variance(k, earlier.params) + weight * total_variance(k, later.params)

    used = [s for s, w in ((earlier, 1 - weight), (later, weight)) if w > 1e-9]
    low = max(s.k_min for s in used) - RANGE_PAD
    high = min(s.k_max for s in used) + RANGE_PAD
    valid = (k >= low) & (k <= high) & (variance > 0)
    row = np.full(len(k), np.nan)
    row[valid] = np.sqrt(variance[valid] / maturity)
    return row


def surface_iv(slices: List[SviSlice], strikes: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    """
    Implied volatility on a grid of maturities (rows) and strikes (columns), NaN where there is no support.

    Between two fitted expiries, total variance is interpolated linearly in time at the same
    log-moneyness, which keeps it rising with time whenever the fitted expiries do.
    """
    ordered = sorted(slices, key=lambda s: s.maturity)
    times = np.array([s.maturity for s in ordered])
    log_forwards = np.log([s.forward for s in ordered])
    grid = np.full((len(maturities), len(strikes)), np.nan)
    if len(ordered) < 2:
        return grid

    for row, maturity in enumerate(maturities):
        if maturity < times[0] - 1e-12 or maturity > times[-1] + 1e-12:
            continue
        k = np.log(strikes / np.exp(np.interp(maturity, times, log_forwards)))
        grid[row] = _blend_row(ordered, times, maturity, k)
    return grid


def surface_iv_log_moneyness(slices: List[SviSlice], k_grid: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    """The same surface on a grid of maturities (rows) and log-moneyness values k (columns)."""
    ordered = sorted(slices, key=lambda s: s.maturity)
    times = np.array([s.maturity for s in ordered])
    grid = np.full((len(maturities), len(k_grid)), np.nan)
    if len(ordered) < 2:
        return grid

    for row, maturity in enumerate(maturities):
        if maturity < times[0] - 1e-12 or maturity > times[-1] + 1e-12:
            continue
        grid[row] = _blend_row(ordered, times, maturity, k_grid)
    return grid


@dataclass(frozen=True)
class BlendedSmile:
    """The SVI smile at a maturity between two fitted expiries; its k range is covered by both fits."""
    earlier: SviSlice
    later: SviSlice
    weight: float
    forward: float
    k_low: float
    k_high: float

    def variance(self, k: np.ndarray) -> np.ndarray:
        """Total variance at log-moneyness k, blended linearly in time between the two expiries."""
        return ((1 - self.weight) * total_variance(k, self.earlier.params)
                + self.weight * total_variance(k, self.later.params))


def blended_smile(earlier: SviSlice, later: SviSlice, maturity: float) -> BlendedSmile:
    """Blend the fits of two expiries into a smile for a maturity between them."""
    span = later.maturity - earlier.maturity
    weight = float(np.clip((maturity - earlier.maturity) / span, 0.0, 1.0)) if span > 0 else 0.0
    used = [s for s, w in ((earlier, 1 - weight), (later, weight)) if w > 1e-9]
    forward = float(np.exp((1 - weight) * np.log(earlier.forward) + weight * np.log(later.forward)))
    return BlendedSmile(earlier, later, weight, forward,
                        max(s.k_min for s in used), min(s.k_max for s in used))


def _delta_gap(k: float, smile: BlendedSmile, maturity: float, dividend_yield: float, delta: float) -> float:
    """The Black-Scholes spot delta at log-moneyness k, minus the target delta."""
    total = float(smile.variance(np.array([k]))[0])
    d1 = (-k + total / 2) / np.sqrt(total)
    discount = np.exp(-dividend_yield * maturity)
    spot_delta = discount * norm.cdf(d1) if delta > 0 else discount * (norm.cdf(d1) - 1)
    return float(spot_delta - delta)


def delta_log_moneyness(smile: BlendedSmile, maturity: float, dividend_yield: float,
                        delta: float) -> Optional[float]:
    """The log-moneyness k where a Black-Scholes spot delta equals ``delta`` (positive for calls, negative for puts)."""
    arguments = (smile, maturity, dividend_yield, delta)
    if _delta_gap(smile.k_low, *arguments) * _delta_gap(smile.k_high, *arguments) > 0:
        return None
    return float(brentq(_delta_gap, smile.k_low, smile.k_high, args=arguments))