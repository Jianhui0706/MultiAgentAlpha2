# Core Rules

## Dataset Features

- Features must come from the real dataset schema.
- Raw features must use the `$feature_name` format.
- Do not invent fields that are not present in `domain_dataset_summary.feature_columns`.
- Do not guess horizon-specific aliases such as `$return`, `$returns`, or `$volume_change_4h` unless they are explicitly present in the schema.
- In this project, `$volume_change_4h` and `$volume_change_12h` are common illegal hallucinations; do not use them unless they appear in the runtime whitelist.
- Do not use `fwd_*` columns or any future-looking target fields in factor expressions.

## Functions

- Use only supported runtime functions.
- Do not invent helper functions, aliases, or pseudo-functions.
- Respect the intended difference between cross-sectional functions and time-series functions.
- Respect function arity exactly; if a window argument is required, provide it.
- If you mean a rolling statistic, use the `TS_*` form instead of guessing a cross-sectional shortcut.
- `TS_PERCENTILE` is not part of the supported runtime set here; do not use it as a threshold shortcut.

## Expression Safety

- Prefer simple expressions over deep nesting unless the extra structure is necessary.
- Add `+1e-8` to denominators when division can become unstable.
- Do not output placeholder text, pseudocode, or natural-language formulas as expressions.
- If a function-window form is required, use the proper time-series function rather than guessing a shorthand.
- Do not submit an expression that still contains known legality issues; repair it first if the fix is local and obvious.

## Repair Behavior

- If validation feedback points to a specific symbol, function, or arity issue, repair that issue first.
- Do not silently switch to unrelated features or a different hypothesis when a local repair is possible.
- Prefer local legality repair over hypothesis drift:
  - fix missing `$` prefixes
  - replace unsupported rolling aliases with supported `TS_*` functions
  - remove invented features instead of renaming them to another invented feature
