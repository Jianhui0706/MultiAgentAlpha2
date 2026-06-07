# STD

## Purpose

`STD(A)` is a cross-sectional standard deviation function.

## Arity

- Expected arguments: 1

## Use

- Use when the intent is a current-time cross-sectional dispersion transform.

## Example

- `STD($returns_24h)`

## Common Mistakes

- Do not write `STD(A, n)` for a rolling standard deviation.
- If the goal is a time-window volatility measure, use `TS_STD(A, n)` instead.
