from __future__ import annotations

import os
import re
import warnings
import numpy as np
import pandas as pd
from typing import Any

# 环境变量 FACTOR_USE_ORIGINAL=1 可强制使用原始 pandas 版本（调试/对比用）
_USE_MATRIX_DEFAULT = not os.getenv("FACTOR_USE_ORIGINAL", "").strip().lower() in ("1", "true", "yes")
_VERBOSE_RUNTIME = os.getenv("FACTOR_RUNTIME_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}


def _runtime_print(message: str) -> None:
    if _VERBOSE_RUNTIME:
        print(message)


def RANK(series: pd.Series) -> pd.Series:
    """Cross-sectional ranking (percentile rank within each datetime).
    
    Handles edge cases:
    - When all values are the same, returns 0.5 (middle rank)
    - When values are NaN, returns NaN
    """
    def safe_rank(x):
        if x.isna().all():
            return x
        if x.std() < 1e-10:
            return pd.Series(0.5, index=x.index)
        return x.rank(pct=True)
    
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="datetime").transform(safe_rank)
    return safe_rank(series)


def ZSCORE(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="datetime").transform(lambda x: (x - x.mean()) / (x.std() + 1e-10))
    return (series - series.mean()) / (series.std() + 1e-10)


def MEAN(series: pd.Series) -> pd.Series:
    """Cross-sectional mean."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="datetime").transform("mean")
    return pd.Series([series.mean()] * len(series), index=series.index)


def STD(series: pd.Series) -> pd.Series:
    """Cross-sectional standard deviation."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="datetime").transform("std")
    return pd.Series([series.std()] * len(series), index=series.index)


def DELTA(series: pd.Series, n: int) -> pd.Series:
    """Change in value over n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").diff(n)
    return series.diff(n)


def DELAY(series: pd.Series, n: int) -> pd.Series:
    """Value delayed by n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").shift(n)
    return series.shift(n)


def TS_MEAN(series: pd.Series, n: int) -> pd.Series:
    """Rolling mean over past n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).mean())
    return series.rolling(window=n, min_periods=1).mean()


def TS_SUM(series: pd.Series, n: int) -> pd.Series:
    """Rolling sum over past n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).sum())
    return series.rolling(window=n, min_periods=1).sum()


def TS_RANK(series: pd.Series, n: int) -> pd.Series:
    """Time-series rank of the last value in the past n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).rank(pct=True))
    return series.rolling(window=n, min_periods=1).rank(pct=True)


def TS_ZSCORE(series: pd.Series, n: int) -> pd.Series:
    """Rolling z-score over past n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(
            lambda x: (x - x.rolling(window=n, min_periods=1).mean()) / (x.rolling(window=n, min_periods=1).std() + 1e-10)
        )
    return (series - series.rolling(window=n, min_periods=1).mean()) / (series.rolling(window=n, min_periods=1).std() + 1e-10)


def TS_MEDIAN(series: pd.Series, n: int) -> pd.Series:
    """Rolling median over past n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).median())
    return series.rolling(window=n, min_periods=1).median()


def TS_PCTCHANGE(series: pd.Series, n: int) -> pd.Series:
    """Percentage change over n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(lambda x: x.pct_change(periods=n))
    return series.pct_change(periods=n)


def TS_MIN(series: pd.Series, n: int) -> pd.Series:
    """Rolling minimum over past n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).min())
    return series.rolling(window=n, min_periods=1).min()


def TS_MAX(series: pd.Series, n: int) -> pd.Series:
    """Rolling maximum over past n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).max())
    return series.rolling(window=n, min_periods=1).max()


def TS_STD(series: pd.Series, n: int) -> pd.Series:
    """Rolling standard deviation over past n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).std())
    return series.rolling(window=n, min_periods=1).std()


def TS_VAR(series: pd.Series, n: int) -> pd.Series:
    """Rolling variance over past n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).var())
    return series.rolling(window=n, min_periods=1).var()


def TS_CORR(series1: pd.Series, series2: pd.Series, n: int) -> pd.Series:
    """Rolling correlation between two series over past n periods."""
    if isinstance(series1.index, pd.MultiIndex):
        def rolling_corr(group):
            df = pd.DataFrame({"a": group, "b": series2.loc[group.index]})
            return df["a"].rolling(window=n, min_periods=1).corr(df["b"])
        return series1.groupby(level="symbol").transform(rolling_corr)
    return series1.rolling(window=n, min_periods=1).corr(series2)


def TS_COV(series1: pd.Series, series2: pd.Series, n: int) -> pd.Series:
    """Rolling covariance between two series over past n periods."""
    if isinstance(series1.index, pd.MultiIndex):
        def rolling_cov(group):
            df = pd.DataFrame({"a": group, "b": series2.loc[group.index]})
            return df["a"].rolling(window=n, min_periods=1).cov(df["b"])
        return series1.groupby(level="symbol").transform(rolling_cov)
    return series1.rolling(window=n, min_periods=1).cov(series2)


def TS_RATIO(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Element-wise ratio with numerical stabilization."""
    aligned_1, aligned_2 = series1.align(series2, join="outer")
    return aligned_1 / (aligned_2 + 1e-10)


def ABS(series: pd.Series) -> pd.Series:
    """Absolute value."""
    return series.abs()


def LOG(series: pd.Series) -> pd.Series:
    """Natural logarithm."""
    return np.log(series.abs() + 1e-10)


def SIGN(series: pd.Series) -> pd.Series:
    """Sign of values."""
    return np.sign(series)


def SQRT(series: pd.Series) -> pd.Series:
    """Square root."""
    return np.sqrt(series.abs())


def MAX(series: pd.Series) -> pd.Series:
    """Cross-sectional maximum."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="datetime").transform("max")
    return pd.Series([series.max()] * len(series), index=series.index)


def MIN(series: pd.Series) -> pd.Series:
    """Cross-sectional minimum."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="datetime").transform("min")
    return pd.Series([series.min()] * len(series), index=series.index)


def MEDIAN(series: pd.Series) -> pd.Series:
    """Cross-sectional median."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="datetime").transform("median")
    return pd.Series([series.median()] * len(series), index=series.index)


def SKEW(series: pd.Series) -> pd.Series:
    """Cross-sectional skewness."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="datetime").transform(lambda x: x.skew())
    return pd.Series([series.skew()] * len(series), index=series.index)


def KURT(series: pd.Series) -> pd.Series:
    """Cross-sectional kurtosis."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="datetime").transform(lambda x: x.kurtosis())
    return pd.Series([series.kurtosis()] * len(series), index=series.index)


def RSI(series: pd.Series, n: int) -> pd.Series:
    """Relative Strength Index over n periods."""
    if isinstance(series.index, pd.MultiIndex):
        delta = series.groupby(level="symbol").diff()
    else:
        delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    if isinstance(series.index, pd.MultiIndex):
        avg_gain = gain.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).mean())
        avg_loss = loss.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).mean())
    else:
        avg_gain = gain.rolling(window=n, min_periods=1).mean()
        avg_loss = loss.rolling(window=n, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def BB_UPPER(series: pd.Series, n: int, num_std: float = 2.0) -> pd.Series:
    """Upper Bollinger Band."""
    if isinstance(series.index, pd.MultiIndex):
        middle = series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).mean())
        std = series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).std())
    else:
        middle = series.rolling(window=n, min_periods=1).mean()
        std = series.rolling(window=n, min_periods=1).std()
    return middle + num_std * std


def BB_LOWER(series: pd.Series, n: int, num_std: float = 2.0) -> pd.Series:
    """Lower Bollinger Band."""
    if isinstance(series.index, pd.MultiIndex):
        middle = series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).mean())
        std = series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).std())
    else:
        middle = series.rolling(window=n, min_periods=1).mean()
        std = series.rolling(window=n, min_periods=1).std()
    return middle - num_std * std


def BB_MIDDLE(series: pd.Series, n: int) -> pd.Series:
    """Middle Bollinger Band (SMA)."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(lambda x: x.rolling(window=n, min_periods=1).mean())
    return series.rolling(window=n, min_periods=1).mean()


def MACD(series: pd.Series, short_period: int = 12, long_period: int = 26) -> pd.Series:
    """Moving Average Convergence Divergence."""
    if isinstance(series.index, pd.MultiIndex):
        ema_short = series.groupby(level="symbol").transform(
            lambda x: x.ewm(span=short_period, adjust=False).mean()
        )
        ema_long = series.groupby(level="symbol").transform(
            lambda x: x.ewm(span=long_period, adjust=False).mean()
        )
    else:
        ema_short = series.ewm(span=short_period, adjust=False).mean()
        ema_long = series.ewm(span=long_period, adjust=False).mean()
    return ema_short - ema_long


def EXP(series: pd.Series) -> pd.Series:
    """Exponential."""
    return np.exp(series)


def POW(series: pd.Series, n: float) -> pd.Series:
    """Power function."""
    return np.power(series, n)


def PERCENTILE(series: pd.Series, q: float, n: int) -> pd.Series:
    """Rolling quantile over past n periods at percentile q (0-1)."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(
            lambda x: x.rolling(window=n, min_periods=1).quantile(q)
        )
    return series.rolling(window=n, min_periods=1).quantile(q)


def TS_QUANTILE(series: pd.Series, n: int, q: float) -> pd.Series:
    """Rolling quantile over past n periods at quantile q (0-1). Alias for PERCENTILE."""
    return PERCENTILE(series, q, n)


def TS_ARGMAX(series: pd.Series, n: int) -> pd.Series:
    """Index (relative to current time) of the maximum value over past n periods.

    ★ 耗时热点：内层 for i in range 逐行遍历，无向量化。O(T * N)。
    """
    def argmax_rolling(x):
        result = np.zeros(len(x))
        for i in range(len(x)):
            window = x.iloc[max(0, i-n+1):i+1]
            if len(window) > 0 and not window.isna().all():
                result[i] = len(window) - 1 - window.values.argmax()
            else:
                result[i] = np.nan
        return pd.Series(result, index=x.index)
    
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(argmax_rolling)
    return argmax_rolling(series)


def TS_ARGMIN(series: pd.Series, n: int) -> pd.Series:
    """Index (relative to current time) of the minimum value over past n periods.

    ★ 耗时热点：同 TS_ARGMAX，逐行循环无向量化。
    """
    def argmin_rolling(x):
        result = np.zeros(len(x))
        for i in range(len(x)):
            window = x.iloc[max(0, i-n+1):i+1]
            if len(window) > 0 and not window.isna().all():
                result[i] = len(window) - 1 - window.values.argmin()
            else:
                result[i] = np.nan
        return pd.Series(result, index=x.index)
    
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(argmin_rolling)
    return argmin_rolling(series)


def TS_MAD(series: pd.Series, n: int) -> pd.Series:
    """Rolling Median Absolute Deviation over past n periods."""
    def mad_rolling(x):
        return x.rolling(window=n, min_periods=1).apply(
            lambda y: np.median(np.abs(y - np.median(y))) if len(y) > 0 else np.nan
        )
    
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(mad_rolling)
    return mad_rolling(series)


def HIGHDAY(series: pd.Series, n: int) -> pd.Series:
    """Number of days since the highest value in past n periods."""
    return TS_ARGMAX(series, n)


def LOWDAY(series: pd.Series, n: int) -> pd.Series:
    """Number of days since the lowest value in past n periods."""
    return TS_ARGMIN(series, n)


def SUMAC(series: pd.Series, n: int) -> pd.Series:
    """Cumulative sum over past n periods."""
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(
            lambda x: x.rolling(window=n, min_periods=1).sum()
        )
    return series.rolling(window=n, min_periods=1).sum()


def SMA(series: pd.Series, n: int, m: float = 1.0) -> pd.Series:
    """Simple moving average with modifier m.
    
    SMA = (m * current + (n-1) * previous_SMA) / n
    """
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(
            lambda x: x.ewm(alpha=m/n, adjust=False).mean()
        )
    return series.ewm(alpha=m/n, adjust=False).mean()


def WMA(series: pd.Series, n: int) -> pd.Series:
    """Weighted moving average with weights decreasing from 0.9 to 0.9^n."""
    def wma_rolling(x):
        weights = np.array([0.9 ** i for i in range(n)])[::-1]
        weights = weights / weights.sum()
        return x.rolling(window=n, min_periods=1).apply(
            lambda y: (y * weights[-len(y):]).sum() / weights[-len(y):].sum() if len(y) > 0 else np.nan
        )
    
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(wma_rolling)
    return wma_rolling(series)


def EMA(series: pd.Series, n: int) -> pd.Series:
    """Exponential moving average with decay factor 2/(n+1)."""
    span = n
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(
            lambda x: x.ewm(span=span, adjust=False).mean()
        )
    return series.ewm(span=span, adjust=False).mean()


def DECAYLINEAR(series: pd.Series, d: int) -> pd.Series:
    """Linearly weighted moving average with weights increasing from 1 to d."""
    def decay_linear_rolling(x):
        weights = np.arange(1, d + 1)
        weights = weights / weights.sum()
        return x.rolling(window=d, min_periods=1).apply(
            lambda y: (y * weights[-len(y):]).sum() / weights[-len(y):].sum() if len(y) > 0 else np.nan
        )
    
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(decay_linear_rolling)
    return decay_linear_rolling(series)


def COUNT(condition: pd.Series, n: int) -> pd.Series:
    """Count of samples satisfying condition in past n periods.
    
    Args:
        condition: Boolean series (e.g., $close > $open)
        n: Number of periods to look back
    """
    if isinstance(condition.index, pd.MultiIndex):
        return condition.groupby(level="symbol").transform(
            lambda x: x.rolling(window=n, min_periods=1).sum()
        )
    return condition.rolling(window=n, min_periods=1).sum()


def SUMIF(series: pd.Series, n: int, condition: pd.Series) -> pd.Series:
    """Sum of series over past n periods where condition is met.
    
    Args:
        series: Values to sum
        n: Number of periods to look back
        condition: Boolean series for filtering
    """
    filtered = series.where(condition, 0.0)
    if isinstance(series.index, pd.MultiIndex):
        return filtered.groupby(level="symbol").transform(
            lambda x: x.rolling(window=n, min_periods=1).sum()
        )
    return filtered.rolling(window=n, min_periods=1).sum()


def FILTER(series: pd.Series, condition: pd.Series) -> pd.Series:
    """Filter series based on condition, setting non-matching values to NaN.
    
    Args:
        series: Values to filter
        condition: Boolean series for filtering
    """
    return series.where(condition, np.nan)


def SEQUENCE(n: int) -> np.ndarray:
    """Generate a sequence from 1 to n.
    
    Used as argument B in REGBETA or REGRESI.
    """
    return np.arange(1, n + 1, dtype=float)


def REGBETA(series_a: pd.Series, series_b: pd.Series | np.ndarray, n: int) -> pd.Series:
    """Rolling regression coefficient of A on B over past n periods.

    ★★★ 最大耗时热点：内层 for i in range 逐行调用 np.linalg.lstsq，
    每行一次矩阵运算 O(K²)。对 500 日 × 100 资产可能需要数分钟。
    优化方向：用 rolling + 批量矩阵运算替代逐行循环。

    Args:
        series_a: Dependent variable (multi-column sequence)
        series_b: Independent variable (single-column or multi-column sequence)
        n: Number of periods for regression
    """
    def rolling_beta(a_vals, b_vals):
        if isinstance(b_vals, np.ndarray):
            b_vals = b_vals[-len(a_vals):]
        else:
            b_vals = b_vals.iloc[-len(a_vals):].values
        
        a_vals = a_vals[-len(b_vals):] if len(a_vals) > len(b_vals) else a_vals
        if len(a_vals) < 2:
            return np.nan
        
        x = b_vals.reshape(-1, 1) if len(b_vals.shape) == 1 else b_vals
        y = a_vals.reshape(-1, 1) if len(a_vals.shape) == 1 else a_vals
        
        try:
            x_with_const = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            return beta[1, 0] if beta.shape[0] > 1 else beta[0, 0]
        except:
            return np.nan
    
    def regbeta_rolling(a_series, b_series_or_array):
        result = np.zeros(len(a_series))
        for i in range(len(a_series)):
            window_a = a_series.iloc[max(0, i-n+1):i+1].values
            if isinstance(b_series_or_array, np.ndarray):
                window_b = b_series_or_array
            else:
                window_b = b_series_or_array.iloc[max(0, i-n+1):i+1].values
            result[i] = rolling_beta(window_a, window_b)
        return pd.Series(result, index=a_series.index)
    
    if isinstance(series_a.index, pd.MultiIndex):
        return series_a.groupby(level="symbol").transform(
            lambda x: regbeta_rolling(x, series_b)
        )
    return regbeta_rolling(series_a, series_b)


def REGRESI(series_a: pd.Series, series_b: pd.Series | np.ndarray, n: int) -> pd.Series:
    """Rolling regression residual of A on B over past n periods.

    ★★★ 最大耗时热点：同 REGBETA，逐行 lstsq 回归。
    优化方向：同 REGBETA。

    Args:
        series_a: Dependent variable (multi-column sequence)
        series_b: Independent variable (single-column or multi-column sequence)
        n: Number of periods for regression
    """
    def rolling_residual(a_vals, b_vals):
        if isinstance(b_vals, np.ndarray):
            b_vals = b_vals[-len(a_vals):]
        else:
            b_vals = b_vals.iloc[-len(a_vals):].values
        
        a_vals = a_vals[-len(b_vals):] if len(a_vals) > len(b_vals) else a_vals
        if len(a_vals) < 2:
            return np.nan
        
        x = b_vals.reshape(-1, 1) if len(b_vals.shape) == 1 else b_vals
        y = a_vals.reshape(-1, 1) if len(a_vals.shape) == 1 else a_vals
        
        try:
            x_with_const = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            y_pred = x_with_const @ beta
            residual = y[-1, 0] - y_pred[-1, 0]
            return residual
        except:
            return np.nan
    
    def regresi_rolling(a_series, b_series_or_array):
        result = np.zeros(len(a_series))
        for i in range(len(a_series)):
            window_a = a_series.iloc[max(0, i-n+1):i+1].values
            if isinstance(b_series_or_array, np.ndarray):
                window_b = b_series_or_array
            else:
                window_b = b_series_or_array.iloc[max(0, i-n+1):i+1].values
            result[i] = rolling_residual(window_a, window_b)
        return pd.Series(result, index=a_series.index)
    
    if isinstance(series_a.index, pd.MultiIndex):
        return series_a.groupby(level="symbol").transform(
            lambda x: regresi_rolling(x, series_b)
        )
    return regresi_rolling(series_a, series_b)


def PROD(series: pd.Series, n: int) -> pd.Series:
    """Product of values over past n periods."""
    def prod_rolling(x):
        return x.rolling(window=n, min_periods=1).apply(
            lambda y: np.prod(y) if len(y) > 0 else np.nan
        )
    
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="symbol").transform(prod_rolling)
    return prod_rolling(series)


def INV(series: pd.Series) -> pd.Series:
    """Reciprocal (1/x) of each element."""
    return 1.0 / (series + 1e-10)


def FLOOR(series: pd.Series) -> pd.Series:
    """Floor of each element."""
    return np.floor(series)


def SCALE(series: pd.Series, target_sum: float = 1.0) -> pd.Series:
    """Scale the absolute values in the cross-section to sum to target_sum."""
    def scale_func(x):
        abs_sum = x.abs().sum()
        if abs_sum < 1e-10:
            return x
        return x * target_sum / abs_sum
    
    if isinstance(series.index, pd.MultiIndex):
        return series.groupby(level="datetime").transform(scale_func)
    return scale_func(series)


def _convert_expression(expression: str) -> str:
    """
    Convert factor expression to valid Python code.
    Replace $variable with _var_variable format.
    """
    # Replace $variable with _var_variable
    converted = re.sub(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', r'_var_\1', expression)
    return converted


def calculate_factor_expression(
    expression: str,
    panel: pd.DataFrame,
    factor_name: str,
    *,
    use_matrix: bool | None = None,
) -> pd.Series | None:
    """Calculate factor values from expression string.

    默认优先使用矩阵化版本（factor_functions_matrix），失败时自动回退到
    原始 pandas groupby 版本。设置环境变量 FACTOR_USE_ORIGINAL=1 或
    传入 use_matrix=False 可强制使用原始版本。

    Args:
        expression: Factor expression like "RANK(TS_PCTCHANGE($close, 6))"
        panel: DataFrame with MultiIndex (datetime, symbol) containing price data
        factor_name: Name of the factor for the result series
        use_matrix: True=矩阵版, False=原始版, None=根据环境变量自动选择

    Returns:
        pd.Series with calculated factor values, or None if calculation fails
    """
    if use_matrix is None:
        use_matrix = _USE_MATRIX_DEFAULT

    if use_matrix:
        try:
            from .factor_functions_matrix import calculate_factor_expression_matrix
            result = calculate_factor_expression_matrix(expression, panel, factor_name)
            if result is not None:
                return result
            # 矩阵版本返回 None 时，回退到原始版本
            _runtime_print(f"[WARN] Matrix version returned None for '{factor_name}', falling back to pandas version")
        except Exception as exc:
            _runtime_print(f"[WARN] Matrix version failed for '{factor_name}': {exc}, falling back to pandas version")

    # ── 原始 pandas groupby 版本（回退路径）──
    try:
        converted_expr = _convert_expression(expression)
        local_vars: dict[str, Any] = {}
        for col in panel.columns:
            local_vars[f"_var_{col}"] = panel[col]
        local_vars.update({
            "RANK": RANK, "ZSCORE": ZSCORE, "MEAN": MEAN, "STD": STD,
            "DELTA": DELTA, "DELAY": DELAY,
            "TS_MEAN": TS_MEAN, "TS_SUM": TS_SUM, "TS_RANK": TS_RANK,
            "TS_ZSCORE": TS_ZSCORE, "TS_MEDIAN": TS_MEDIAN,
            "TS_PCTCHANGE": TS_PCTCHANGE, "TS_MIN": TS_MIN, "TS_MAX": TS_MAX,
            "TS_STD": TS_STD, "TS_VAR": TS_VAR, "TS_CORR": TS_CORR,
            "TS_COV": TS_COV, "TS_RATIO": TS_RATIO,
            "ABS": ABS, "LOG": LOG, "SIGN": SIGN, "SQRT": SQRT,
            "MAX": MAX, "MIN": MIN, "MEDIAN": MEDIAN, "SKEW": SKEW,
            "KURT": KURT, "RSI": RSI,
            "BB_UPPER": BB_UPPER, "BB_LOWER": BB_LOWER, "BB_MIDDLE": BB_MIDDLE,
            "MACD": MACD, "EXP": EXP, "POW": POW,
            "PERCENTILE": PERCENTILE, "TS_QUANTILE": TS_QUANTILE,
            "TS_ARGMAX": TS_ARGMAX, "TS_ARGMIN": TS_ARGMIN, "TS_MAD": TS_MAD,
            "HIGHDAY": HIGHDAY, "LOWDAY": LOWDAY, "SUMAC": SUMAC,
            "SMA": SMA, "WMA": WMA, "EMA": EMA, "DECAYLINEAR": DECAYLINEAR,
            "COUNT": COUNT, "SUMIF": SUMIF, "FILTER": FILTER,
            "SEQUENCE": SEQUENCE, "REGBETA": REGBETA, "REGRESI": REGRESI,
            "PROD": PROD, "INV": INV, "FLOOR": FLOOR, "SCALE": SCALE,
            "np": np, "pd": pd,
        })
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*invalid value encountered.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*divide by zero encountered.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*Mean of empty slice.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*Degrees of freedom <= 0.*")
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                result = eval(converted_expr, {"__builtins__": {}}, local_vars)
        if isinstance(result, pd.Series):
            return result.rename(factor_name)
        elif isinstance(result, pd.DataFrame):
            return result.iloc[:, 0].rename(factor_name)
        else:
            return pd.Series(result, index=panel.index).rename(factor_name)
    except Exception as e:
        _runtime_print(f"[ERROR] Failed to calculate factor '{factor_name}': {e}")
        _runtime_print(f"[ERROR] Original expression: {expression}")
        _runtime_print(f"[ERROR] Converted expression: {converted_expr if 'converted_expr' in dir() else 'N/A'}")
        return None
