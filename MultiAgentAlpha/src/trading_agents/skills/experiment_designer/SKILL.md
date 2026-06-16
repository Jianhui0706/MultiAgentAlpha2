# ExperimentDesigner Rule Pack

Use this rule pack when generating or repairing factor expressions for `ExperimentDesignerAgent`.

## Workflow

1. Read the available dataset schema first.
2. Build expressions only from real dataset features.
3. Use only supported runtime functions.
4. Prefer simple, interpretable expressions before complex nesting.
5. Run a strict pre-submit legality check before finalizing any factor.
6. Normalize common rolling-function confusion before finalizing an expression.
7. Return only expressions that are ready for local validation.
8. If validation feedback is available, repair the expression before exploring a different idea.

## First-Pass Legality Priorities

Before returning a factor, confirm all of the following:

1. Every raw feature in `expression` is present in the runtime dataset schema.
2. Every raw feature uses the `$feature_name` form.
3. Every function is part of the runtime-supported function set.
4. Every function call uses the correct number of arguments.
5. No `fwd_*` field appears in `expression` or `variables`.
6. `variables` contains only the dataset features actually used in `expression`.
7. The final answer is one strict JSON object with no markdown or prose wrapper.

## High-Risk Illegal Outputs To Avoid

Do not emit these unless the runtime whitelist explicitly includes them:

- `$volume_change_4h`
- `$volume_change_12h`
- `$return`
- `$returns`

Do not emit unsupported functions such as:

- `TS_PERCENTILE`

## Intent

This rule pack improves first-pass expression quality. It does not replace local code validators.
