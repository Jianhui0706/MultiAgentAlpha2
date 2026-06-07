# DELAY

## Purpose

`DELAY(A, n)` shifts a series backward by `n` periods.

## Arity

- Expected arguments: 2

## Use

- Use when comparing a current value to a prior value from `n` periods ago.

## Example

- `DELAY($close, 4)`

## Common Mistakes

- Do not omit the lag argument.
- Do not use `DELAY` as a future-looking operator.
