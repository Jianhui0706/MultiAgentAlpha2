# RANK

## Purpose

`RANK(A)` is a cross-sectional ranking function.

## Arity

- Expected arguments: 1

## Use

- Use when you want relative cross-sectional ordering at the current timestamp.

## Example

- `RANK($close)`

## Common Mistakes

- Do not pass a window length.
- Do not use `RANK(A, n)`.
