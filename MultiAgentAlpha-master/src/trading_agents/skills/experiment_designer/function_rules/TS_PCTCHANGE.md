# TS_PCTCHANGE

## Purpose

`TS_PCTCHANGE(A, n)` measures time-series percentage change over a lookback window.

## Arity

- Expected arguments: 2

## Use

- Use when you need a rolling return-like change over `n` periods.

## Example

- `TS_PCTCHANGE($close, 4)`

## Common Mistakes

- Do not omit the window argument.
- Do not use a cross-sectional function when the intent is temporal change.
