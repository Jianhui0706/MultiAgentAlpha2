# Output Contract

## Required Format

Return only one strict JSON object.

- Do not return markdown.
- Do not return fenced code blocks.
- Do not return explanatory text before or after the JSON object.
- Do not return XML tags, `<think>` tags, comments, or analysis text.

## Required Factor Fields

Each factor entry must provide:

- `factor_name`
- `description`
- `formulation`
- `expression`
- `variables`

## Exact Schema

Use `factors` as a JSON array, not a dictionary keyed by factor name.

```json
{
  "task_plan": ["short step 1", "short step 2"],
  "factors": [
    {
      "factor_name": "snake_case_name",
      "description": "one short sentence",
      "formulation": "plain text formula, no LaTeX braces",
      "expression": "RANK(TS_PCTCHANGE($close, 4))",
      "variables": {"$close": "close price"},
      "theory_alignment": "mainline or counterfactual",
      "counterfactual_thesis": "only for counterfactual",
      "counterfactual_mechanism": "only for counterfactual",
      "counterfactual_critique": "only for counterfactual",
      "counterfactual_expected_direction": "only for counterfactual"
    }
  ]
}
```

## Task Plan

- Provide a short 2-4 step construction plan.
- Each step should explain what is being built and why.

## Variables Field

- `variables` must only contain dataset features actually used in the expression.
- Do not list unused variables.
- Do not include invented features.
- Use the exact `$feature_name` spelling that appears in the runtime dataset schema.

## Expression Field

- `expression` must be a concrete runtime-ready expression string.
- Do not return placeholders such as `TBD`, `...`, or pseudocode.
- All string values must be quoted and escaped.
- Do not use single quotes for JSON strings.
- Do not use LaTeX-only syntax such as `{Z(...)}` in `formulation`.
- Do not use bare feature names like `close`; write `$close`.
- Do not use unsupported shortcuts such as `$volume_change_4h`, `$volume_change_12h`, or `TS_PERCENTILE` unless they are explicitly present in the runtime whitelist.

## Pre-Submit Checklist

Before returning the final JSON:

1. Ensure every factor has a non-empty `factor_name`, `description`, `formulation`, `expression`, and `variables`.
2. Ensure every raw feature in `expression` appears in `variables`.
3. Ensure `variables` does not mention any feature absent from `expression`.
4. Ensure every function call is runtime-supported and uses the correct arity.
5. Ensure the final response starts with `{` and ends with `}`.
