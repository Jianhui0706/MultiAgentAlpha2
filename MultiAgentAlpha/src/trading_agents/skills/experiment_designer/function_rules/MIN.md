# MIN

## Purpose

`MIN(A)` is a cross-sectional minimum function.

## Arity

- Expected arguments: 1

## Use

- Use when the intent is a current-time cross-sectional minimum transformation.

## Example

- `MIN($close)`

## Common Mistakes

- Do not write `MIN(A, n)` when the intent is a rolling minimum.
- If the goal is a recent-window minimum, use `TS_MIN(A, n)` instead.
