# TS_STD

## Purpose

`TS_STD(A, n)` computes a rolling standard deviation over time.

## Arity

- Expected arguments: 2

## Use

- Use for time-series volatility-like behavior over `n` periods.

## Example

- `TS_STD($close, 24)`

## Common Mistakes

- Do not write `STD(A, n)` when the intent is a rolling statistic.
- `STD(A)` and `TS_STD(A, n)` are not the same.
