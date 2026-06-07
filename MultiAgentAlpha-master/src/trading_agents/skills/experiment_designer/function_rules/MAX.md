# MAX

## Purpose

`MAX(A)` is a cross-sectional maximum function.

## Arity

- Expected arguments: 1

## Use

- Use when the intent is a current-time cross-sectional maximum transformation.

## Example

- `MAX($close)`

## Common Mistakes

- Do not write `MAX(A, n)` when the intent is a rolling maximum.
- If the goal is a recent-window maximum, use `TS_MAX(A, n)` instead.
