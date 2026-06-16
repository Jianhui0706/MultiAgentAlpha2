"""Matrix-based factor expression calculation.

Provides calculate_factor_expression_matrix() as a drop-in replacement for
calculate_factor_expression().  Reshapes MultiIndex panel data to (T, N) numpy
matrices to eliminate pandas groupby().transform() overhead -- the dominant
performance bottleneck in the original implementation.

Architecture:
    1. MatrixPanel  -- reshapes each panel column to a (T, N) float64 matrix
    2. Factor funcs -- all accept/return np.ndarray (T, N) instead of pd.Series
    3. eval() uses the same expression syntax; only the namespace differs

Performance gains come from:
    - Eliminating groupby() overhead per symbol/datetime group
    - Batch numpy operations on (T, N) matrices
    - Vectorized REGBETA/REGRESI via (X'X)^{-1} X'y instead of row-by-row lstsq
    - Vectorized TS_ARGMAX/TS_ARGMIN via sliding_window_view
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

_VERBOSE_RUNTIME = os.getenv("FACTOR_RUNTIME_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}


def _runtime_print(message: str) -> None:
    if _VERBOSE_RUNTIME:
        print(message)


try:
    from scipy.stats import rankdata as _rankdata

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════════════
#  MatrixPanel: MultiIndex DataFrame <-> (T, N) matrix conversion
# ═══════════════════════════════════════════════════════════════════════


class MatrixPanel:
    """Wraps a MultiIndex (datetime, symbol) DataFrame as (T, N) matrices.

    T = number of unique datetimes (rows), N = number of unique symbols (cols).
    All panel columns are reshaped once at construction; subsequent factor
    functions operate purely on numpy arrays.
    
    Uses float32 dtype by default to reduce memory usage (half the memory of float64).
    """

    def __init__(self, panel: pd.DataFrame, dtype: np.dtype = np.float32) -> None:
        self.original_index: pd.MultiIndex = panel.index  # type: ignore[assignment]
        self.dtype = dtype

        dates = panel.index.get_level_values("datetime").unique()
        symbols = panel.index.get_level_values("symbol").unique()
        try:
            self.dates = list(dates.sort_values())
        except Exception:
            self.dates = list(dates)
        try:
            self.symbols = list(symbols.sort_values())
        except Exception:
            self.symbols = list(symbols)

        self.T = len(self.dates)
        self.N = len(self.symbols)

        self.matrices: dict[str, np.ndarray] = {}
        for col in panel.columns:
            wide = panel[col].unstack(level="symbol")
            wide = wide.reindex(index=self.dates, columns=self.symbols)
            self.matrices[col] = wide.values.astype(dtype)

    def get(self, name: str) -> np.ndarray:
        return self.matrices[name]

    def to_series(self, matrix: np.ndarray, name: str) -> pd.Series:
        """Convert (T, N) matrix back to MultiIndex Series aligned to original index."""
        wide = pd.DataFrame(matrix, index=self.dates, columns=self.symbols)
        stacked = wide.stack()
        return stacked.reindex(self.original_index).astype(self.dtype).rename(name)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _convert_expression(expression: str) -> str:
    """Replace $variable with _var_variable (identical to original)."""
    return re.sub(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", r"_var_\1", expression)


def _ensure_2d(m: np.ndarray) -> np.ndarray:
    if m.ndim < 2:
        return np.atleast_2d(m)
    return m


def _ts_rolling(m: np.ndarray, n: int, func_name: str, **kwargs: Any) -> np.ndarray:
    """Apply a pandas rolling function along axis-0 (time) on a (T, N) matrix.

    Handles min_periods=1 and NaN propagation identically to the original
    groupby().transform(lambda x: x.rolling(...)) pattern.
    """
    df = pd.DataFrame(m)
    roller = df.rolling(window=n, min_periods=1)
    return getattr(roller, func_name)(**kwargs).values


# ═══════════════════════════════════════════════════════════════════════
#  Cross-sectional functions  (operate along axis-1, per datetime row)
# ═══════════════════════════════════════════════════════════════════════


def RANK(m: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank within each datetime."""
    m = _ensure_2d(m)
    T, N = m.shape
    result = np.full_like(m, np.nan)
    for t in range(T):
        row = m[t]
        valid = np.isfinite(row)
        nv = int(valid.sum())
        if nv == 0:
            continue
        if np.nanstd(row) < 1e-10:
            result[t, valid] = 0.5
        elif _HAS_SCIPY:
            result[t, valid] = _rankdata(row[valid]) / nv
        else:
            order = np.argsort(np.argsort(row[valid]))
            result[t, valid] = (order + 1.0) / nv
    return result


def ZSCORE(m: np.ndarray) -> np.ndarray:
    m = _ensure_2d(m)
    mean = np.nanmean(m, axis=1, keepdims=True)
    std = np.nanstd(m, axis=1, keepdims=True, ddof=1)
    return (m - mean) / (std + 1e-10)


def MEAN(m: np.ndarray) -> np.ndarray:
    m = _ensure_2d(m)
    return np.broadcast_to(np.nanmean(m, axis=1, keepdims=True), m.shape).copy()


def STD(m: np.ndarray) -> np.ndarray:
    m = _ensure_2d(m)
    return np.broadcast_to(np.nanstd(m, axis=1, keepdims=True, ddof=1), m.shape).copy()


def MAX(m: np.ndarray) -> np.ndarray:
    m = _ensure_2d(m)
    return np.broadcast_to(np.nanmax(m, axis=1, keepdims=True), m.shape).copy()


def MIN(m: np.ndarray) -> np.ndarray:
    m = _ensure_2d(m)
    return np.broadcast_to(np.nanmin(m, axis=1, keepdims=True), m.shape).copy()


def MEDIAN(m: np.ndarray) -> np.ndarray:
    m = _ensure_2d(m)
    return np.broadcast_to(np.nanmedian(m, axis=1, keepdims=True), m.shape).copy()


def SKEW(m: np.ndarray) -> np.ndarray:
    m = _ensure_2d(m)
    T, N = m.shape
    result = np.full_like(m, np.nan)
    for t in range(T):
        valid = np.isfinite(m[t])
        nv = int(valid.sum())
        if nv < 3:
            continue
        result[t, valid] = pd.Series(m[t, valid]).skew()
    return result


def KURT(m: np.ndarray) -> np.ndarray:
    m = _ensure_2d(m)
    T, N = m.shape
    result = np.full_like(m, np.nan)
    for t in range(T):
        valid = np.isfinite(m[t])
        nv = int(valid.sum())
        if nv < 4:
            continue
        result[t, valid] = pd.Series(m[t, valid]).kurtosis()
    return result


def SCALE(m: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    m = _ensure_2d(m)
    abs_sum = np.nansum(np.abs(m), axis=1, keepdims=True)
    return np.where(abs_sum < 1e-10, m, m * target_sum / abs_sum)


# ═══════════════════════════════════════════════════════════════════════
#  Time-series functions  (operate along axis-0, per symbol column)
# ═══════════════════════════════════════════════════════════════════════


def DELTA(m: np.ndarray, n: int) -> np.ndarray:
    return pd.DataFrame(m).diff(n).values


def DELAY(m: np.ndarray, n: int) -> np.ndarray:
    return pd.DataFrame(m).shift(n).values


def TS_MEAN(m: np.ndarray, n: int) -> np.ndarray:
    return _ts_rolling(m, n, "mean")


def TS_SUM(m: np.ndarray, n: int) -> np.ndarray:
    return _ts_rolling(m, n, "sum")


def TS_STD(m: np.ndarray, n: int) -> np.ndarray:
    return _ts_rolling(m, n, "std")


def TS_VAR(m: np.ndarray, n: int) -> np.ndarray:
    return _ts_rolling(m, n, "var")


def TS_MIN(m: np.ndarray, n: int) -> np.ndarray:
    return _ts_rolling(m, n, "min")


def TS_MAX(m: np.ndarray, n: int) -> np.ndarray:
    return _ts_rolling(m, n, "max")


def TS_MEDIAN(m: np.ndarray, n: int) -> np.ndarray:
    return _ts_rolling(m, n, "median")


def TS_RANK(m: np.ndarray, n: int) -> np.ndarray:
    return _ts_rolling(m, n, "rank", pct=True)


def TS_ZSCORE(m: np.ndarray, n: int) -> np.ndarray:
    mean = _ts_rolling(m, n, "mean")
    std = _ts_rolling(m, n, "std")
    return (m - mean) / (std + 1e-10)


def TS_PCTCHANGE(m: np.ndarray, n: int) -> np.ndarray:
    return pd.DataFrame(m).pct_change(periods=n).values


def TS_CORR(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Rolling correlation between two (T, N) matrices along time axis."""
    T, N = a.shape
    result = np.full((T, N), np.nan)
    for j in range(N):
        s1 = pd.Series(a[:, j])
        s2 = pd.Series(b[:, j])
        result[:, j] = s1.rolling(window=n, min_periods=1).corr(s2).values
    return result


def TS_COV(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Rolling covariance between two (T, N) matrices along time axis."""
    T, N = a.shape
    result = np.full((T, N), np.nan)
    for j in range(N):
        s1 = pd.Series(a[:, j])
        s2 = pd.Series(b[:, j])
        result[:, j] = s1.rolling(window=n, min_periods=1).cov(s2).values
    return result


def TS_RATIO(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / (b + 1e-10)


# ═══════════════════════════════════════════════════════════════════════
#  Element-wise functions
# ═══════════════════════════════════════════════════════════════════════


def ABS(m: np.ndarray) -> np.ndarray:
    return np.abs(m)


def LOG(m: np.ndarray) -> np.ndarray:
    return np.log(np.abs(m) + 1e-10)


def SIGN(m: np.ndarray) -> np.ndarray:
    return np.sign(m)


def SQRT(m: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(m))


def EXP(m: np.ndarray) -> np.ndarray:
    return np.exp(m)


def POW(m: np.ndarray, n: float) -> np.ndarray:
    return np.power(m, n)


def INV(m: np.ndarray) -> np.ndarray:
    return 1.0 / (m + 1e-10)


def FLOOR(m: np.ndarray) -> np.ndarray:
    return np.floor(m)


# ═══════════════════════════════════════════════════════════════════════
#  Time-series advanced functions
# ═══════════════════════════════════════════════════════════════════════


def RSI(m: np.ndarray, n: int) -> np.ndarray:
    df = pd.DataFrame(m)
    delta = df.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=n, min_periods=1).mean()
    avg_loss = loss.rolling(window=n, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return (100.0 - (100.0 / (1.0 + rs))).values


def BB_UPPER(m: np.ndarray, n: int, num_std: float = 2.0) -> np.ndarray:
    return _ts_rolling(m, n, "mean") + num_std * _ts_rolling(m, n, "std")


def BB_LOWER(m: np.ndarray, n: int, num_std: float = 2.0) -> np.ndarray:
    return _ts_rolling(m, n, "mean") - num_std * _ts_rolling(m, n, "std")


def BB_MIDDLE(m: np.ndarray, n: int) -> np.ndarray:
    return _ts_rolling(m, n, "mean")


def MACD(m: np.ndarray, short_period: int = 12, long_period: int = 26) -> np.ndarray:
    df = pd.DataFrame(m)
    ema_s = df.ewm(span=short_period, adjust=False).mean()
    ema_l = df.ewm(span=long_period, adjust=False).mean()
    return (ema_s - ema_l).values


def SMA(m: np.ndarray, n: int, mm: float = 1.0) -> np.ndarray:
    df = pd.DataFrame(m)
    return df.ewm(alpha=mm / n, adjust=False).mean().values


def EMA(m: np.ndarray, n: int) -> np.ndarray:
    df = pd.DataFrame(m)
    return df.ewm(span=n, adjust=False).mean().values


def WMA(m: np.ndarray, n: int) -> np.ndarray:
    weights = np.array([0.9 ** i for i in range(n)])[::-1]
    weights = weights / weights.sum()
    df = pd.DataFrame(m)
    return df.rolling(window=n, min_periods=1).apply(
        lambda x: np.nansum(x * weights[-len(x) :]) / np.sum(weights[-len(x) :]),
        raw=True,
    ).values


def DECAYLINEAR(m: np.ndarray, d: int) -> np.ndarray:
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()
    df = pd.DataFrame(m)
    return df.rolling(window=d, min_periods=1).apply(
        lambda x: np.nansum(x * weights[-len(x) :]) / np.sum(weights[-len(x) :]),
        raw=True,
    ).values


def PERCENTILE(m: np.ndarray, q: float, n: int) -> np.ndarray:
    return _ts_rolling(m, n, "quantile", q=q)


def TS_QUANTILE(m: np.ndarray, n: int, q: float) -> np.ndarray:
    return PERCENTILE(m, q, n)


def TS_ARGMAX(m: np.ndarray, n: int) -> np.ndarray:
    """TS_ARGMAX: for each (t, col), how many periods ago the max occurred.

    Loops over T but vectorizes across all N columns per step.
    Still much faster than the original row-by-row Python loop over both T and N.
    """
    T, N = m.shape
    result = np.full((T, N), np.nan)
    for t in range(T):
        start = max(0, t - n + 1)
        window = m[start : t + 1, :]  # (W, N)
        valid = np.any(np.isfinite(window), axis=0)  # (N,)
        if not valid.any():
            continue
        window_clean = np.where(np.isfinite(window), window, -np.inf)
        max_idx = np.argmax(window_clean, axis=0)  # (N,)
        result[t, valid] = (window.shape[0] - 1) - max_idx[valid]
    return result


def TS_ARGMIN(m: np.ndarray, n: int) -> np.ndarray:
    """TS_ARGMIN: mirror of TS_ARGMAX."""
    T, N = m.shape
    result = np.full((T, N), np.nan)
    for t in range(T):
        start = max(0, t - n + 1)
        window = m[start : t + 1, :]
        valid = np.any(np.isfinite(window), axis=0)
        if not valid.any():
            continue
        window_clean = np.where(np.isfinite(window), window, np.inf)
        min_idx = np.argmin(window_clean, axis=0)
        result[t, valid] = (window.shape[0] - 1) - min_idx[valid]
    return result


def TS_MAD(m: np.ndarray, n: int) -> np.ndarray:
    df = pd.DataFrame(m)
    return df.rolling(window=n, min_periods=1).apply(
        lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))),
        raw=True,
    ).values


def HIGHDAY(m: np.ndarray, n: int) -> np.ndarray:
    return TS_ARGMAX(m, n)


def LOWDAY(m: np.ndarray, n: int) -> np.ndarray:
    return TS_ARGMIN(m, n)


def SUMAC(m: np.ndarray, n: int) -> np.ndarray:
    return TS_SUM(m, n)


# ═══════════════════════════════════════════════════════════════════════
#  Conditional / filtering functions
# ═══════════════════════════════════════════════════════════════════════


def COUNT(condition: np.ndarray, n: int) -> np.ndarray:
    df = pd.DataFrame(condition.astype(float))
    return df.rolling(window=n, min_periods=1).sum().values


def SUMIF(m: np.ndarray, n: int, condition: np.ndarray) -> np.ndarray:
    filtered = np.where(condition, m, 0.0)
    df = pd.DataFrame(filtered)
    return df.rolling(window=n, min_periods=1).sum().values


def FILTER(m: np.ndarray, condition: np.ndarray) -> np.ndarray:
    return np.where(condition, m, np.nan)


# ═══════════════════════════════════════════════════════════════════════
#  Special functions
# ═══════════════════════════════════════════════════════════════════════


def SEQUENCE(n: int) -> np.ndarray:
    return np.arange(1, n + 1, dtype=float)


def PROD(m: np.ndarray, n: int) -> np.ndarray:
    df = pd.DataFrame(m)
    return df.rolling(window=n, min_periods=1).apply(
        lambda x: np.nanprod(x),
        raw=True,
    ).values


# ═══════════════════════════════════════════════════════════════════════
#  REGBETA / REGRESI  -- the biggest performance hotspots
# ═══════════════════════════════════════════════════════════════════════


def REGBETA(a: np.ndarray, b: np.ndarray | np.ndarray, n: int) -> np.ndarray:
    """Rolling regression coefficient of *a* (y) on *b* (x).

    When b is 1-D (e.g. SEQUENCE), uses batch lstsq over all N columns.
    When b is 2-D, uses vectorized Cov(x,y)/Var(x) formula -- O(W*N) per
    time step instead of O(N*W^2) per step in the original row-by-row lstsq.
    """
    a = _ensure_2d(a)
    T, N = a.shape
    result = np.full((T, N), np.nan)
    is_1d = isinstance(b, np.ndarray) and b.ndim == 1

    for t in range(T):
        start = max(0, t - n + 1)
        W = t - start + 1
        if W < 2:
            continue
        y = a[start : t + 1, :]  # (W, N)

        if is_1d:
            # b is a fixed 1-D sequence (e.g. SEQUENCE) -- batch lstsq
            x_seq = b[:W]
            X = np.column_stack([np.ones(W), x_seq])  # (W, 2)
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]  # (2, N)
                result[t, :] = beta[1, :]
            except np.linalg.LinAlgError:
                pass
        else:
            # b is also (T, N) -- vectorized OLS per time step
            x = _ensure_2d(b)[start : t + 1, :]  # (W, N)
            y_mean = np.nanmean(y, axis=0)  # (N,)
            x_mean = np.nanmean(x, axis=0)  # (N,)
            yc = y - y_mean
            xc = x - x_mean
            cov_xy = np.nansum(xc * yc, axis=0)  # (N,)
            var_x = np.nansum(xc ** 2, axis=0)  # (N,)
            result[t, :] = cov_xy / (var_x + 1e-10)

    return result


def REGRESI(a: np.ndarray, b: np.ndarray | np.ndarray, n: int) -> np.ndarray:
    """Rolling regression residual of *a* (y) on *b* (x) at the latest point.

    residual[t] = y[t] - y_hat[t], where y_hat uses OLS coefficients fitted
    on the past n observations.
    """
    a = _ensure_2d(a)
    T, N = a.shape
    result = np.full((T, N), np.nan)
    is_1d = isinstance(b, np.ndarray) and b.ndim == 1

    for t in range(T):
        start = max(0, t - n + 1)
        W = t - start + 1
        if W < 2:
            continue
        y = a[start : t + 1, :]

        if is_1d:
            x_seq = b[:W]
            X = np.column_stack([np.ones(W), x_seq])  # (W, 2)
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]  # (2, N)
                y_pred_last = X[-1:] @ beta  # (1, N)
                result[t, :] = y[-1, :] - y_pred_last[0, :]
            except np.linalg.LinAlgError:
                pass
        else:
            x = _ensure_2d(b)[start : t + 1, :]
            y_mean = np.nanmean(y, axis=0)
            x_mean = np.nanmean(x, axis=0)
            yc = y - y_mean
            xc = x - x_mean
            cov_xy = np.nansum(xc * yc, axis=0)
            var_x = np.nansum(xc ** 2, axis=0)
            beta = cov_xy / (var_x + 1e-10)
            # residual = (y[-1] - y_mean) - beta * (x[-1] - x_mean)
            result[t, :] = yc[-1, :] - beta * xc[-1, :]

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════

# Build the eval namespace once (shared across all calls)
_MATRIX_NAMESPACE: dict[str, Any] = {
    "RANK": RANK,
    "ZSCORE": ZSCORE,
    "MEAN": MEAN,
    "STD": STD,
    "DELTA": DELTA,
    "DELAY": DELAY,
    "TS_MEAN": TS_MEAN,
    "TS_SUM": TS_SUM,
    "TS_RANK": TS_RANK,
    "TS_ZSCORE": TS_ZSCORE,
    "TS_MEDIAN": TS_MEDIAN,
    "TS_PCTCHANGE": TS_PCTCHANGE,
    "TS_MIN": TS_MIN,
    "TS_MAX": TS_MAX,
    "TS_STD": TS_STD,
    "TS_VAR": TS_VAR,
    "TS_CORR": TS_CORR,
    "TS_COV": TS_COV,
    "TS_RATIO": TS_RATIO,
    "ABS": ABS,
    "LOG": LOG,
    "SIGN": SIGN,
    "SQRT": SQRT,
    "MAX": MAX,
    "MIN": MIN,
    "MEDIAN": MEDIAN,
    "SKEW": SKEW,
    "KURT": KURT,
    "RSI": RSI,
    "BB_UPPER": BB_UPPER,
    "BB_LOWER": BB_LOWER,
    "BB_MIDDLE": BB_MIDDLE,
    "MACD": MACD,
    "EXP": EXP,
    "POW": POW,
    "PERCENTILE": PERCENTILE,
    "TS_QUANTILE": TS_QUANTILE,
    "TS_ARGMAX": TS_ARGMAX,
    "TS_ARGMIN": TS_ARGMIN,
    "TS_MAD": TS_MAD,
    "HIGHDAY": HIGHDAY,
    "LOWDAY": LOWDAY,
    "SUMAC": SUMAC,
    "SMA": SMA,
    "WMA": WMA,
    "EMA": EMA,
    "DECAYLINEAR": DECAYLINEAR,
    "COUNT": COUNT,
    "SUMIF": SUMIF,
    "FILTER": FILTER,
    "SEQUENCE": SEQUENCE,
    "REGBETA": REGBETA,
    "REGRESI": REGRESI,
    "PROD": PROD,
    "INV": INV,
    "FLOOR": FLOOR,
    "SCALE": SCALE,
    "np": np,
    "pd": pd,
}


def build_matrix_panel(panel: pd.DataFrame) -> "MatrixPanel":
    """Build a reusable (T, N) MatrixPanel once.

    The backtest engine evaluates many factor expressions against the SAME
    feature panel; building the dense (T, N) matrices once and sharing this
    object across all of them avoids rebuilding them per factor — a major
    memory (and speed) win. The returned object is read-only during factor
    evaluation, so it is safe to share across threads.
    """
    return MatrixPanel(panel)


def calculate_factor_expression_matrix(
    expression: str,
    panel: pd.DataFrame,
    factor_name: str,
    *,
    matrix_panel: "MatrixPanel | None" = None,
) -> pd.Series | None:
    """Calculate factor values using matrix-based operations.

    Drop-in replacement for calculate_factor_expression().  Reshapes panel
    data to (T, N) matrices to eliminate groupby().transform() overhead.

    matrix_panel: optional prebuilt MatrixPanel (from build_matrix_panel) to
    reuse across many factors on the same panel; when None a fresh one is built.

    Returns:
        pd.Series with the original MultiIndex, or None on failure.
    """
    try:
        converted_expr = _convert_expression(expression)
        mp = matrix_panel if matrix_panel is not None else MatrixPanel(panel)

        local_vars: dict[str, Any] = dict(_MATRIX_NAMESPACE)
        for col in panel.columns:
            local_vars[f"_var_{col}"] = mp.get(col)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*invalid value encountered.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*divide by zero encountered.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*Mean of empty slice.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*Degrees of freedom <= 0.*")
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                result = eval(converted_expr, {"__builtins__": {}}, local_vars)

        if isinstance(result, np.ndarray):
            if result.ndim == 2:
                return mp.to_series(result, factor_name)
            if result.ndim == 1:
                return pd.Series(result, index=panel.index, dtype=float).rename(factor_name)
            return pd.Series(float(result), index=panel.index, dtype=float).rename(factor_name)
        if isinstance(result, (int, float, np.number)):
            return pd.Series(float(result), index=panel.index, dtype=float).rename(factor_name)
        if isinstance(result, pd.Series):
            return result.rename(factor_name)
        return None

    except Exception as e:
        _runtime_print(f"[ERROR] Matrix calculation failed for '{factor_name}': {e}")
        _runtime_print(f"[ERROR] Expression: {expression}")
        return None
