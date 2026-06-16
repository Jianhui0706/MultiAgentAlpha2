# MEAN

## Purpose

`MEAN(A)` is a cross-sectional mean function.

## Arity

- Expected arguments: 1

## Use

- Use when the intent is a current-time cross-sectional averaging transform.

## Example

- `MEAN($volume)`

## Common Mistakes

- Do not write `MEAN(A, n)` for a rolling mean.
- If the goal is a time-window average, use `TS_MEAN(A, n)` instead.
