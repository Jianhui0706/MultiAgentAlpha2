# ZSCORE

## Purpose

`ZSCORE(A)` is a cross-sectional standardization function.

## Arity

- Expected arguments: 1

## Use

- Use when you want cross-sectional normalization at the current timestamp.

## Example

- `ZSCORE($volume)`

## Common Mistakes

- Do not pass a window length.
- Do not confuse `ZSCORE(A)` with `TS_ZSCORE(A, n)`.
