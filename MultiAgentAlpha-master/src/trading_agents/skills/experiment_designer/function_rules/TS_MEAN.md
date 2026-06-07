# TS_MEAN

## Purpose

`TS_MEAN(A, n)` computes a rolling mean over time.

## Arity

- Expected arguments: 2

## Use

- Use for a time-series average over the last `n` periods.

## Example

- `TS_MEAN($volume, 12)`

## Common Mistakes

- Do not write `MEAN(A, n)` when the intent is a rolling mean.
- `MEAN(A)` and `TS_MEAN(A, n)` are not the same.
