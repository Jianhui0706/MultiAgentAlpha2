# TS_MAX

## Purpose

`TS_MAX(A, n)` computes the rolling maximum over the last `n` periods.

## Arity

- Expected arguments: 2

## Use

- Use when the intent is "highest value in the recent window."

## Example

- `TS_MAX($high, 20)`

## Common Mistakes

- Do not write `MAX(A, n)` for a rolling maximum.
- `MAX(A)` and `TS_MAX(A, n)` are not the same.
