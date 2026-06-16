# TS_MIN

## Purpose

`TS_MIN(A, n)` computes the rolling minimum over the last `n` periods.

## Arity

- Expected arguments: 2

## Use

- Use when the intent is "lowest value in the recent window."

## Example

- `TS_MIN($low, 20)`

## Common Mistakes

- Do not write `MIN(A, n)` for a rolling minimum.
- `MIN(A)` and `TS_MIN(A, n)` are not the same.
