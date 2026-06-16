# 20260329 Dingbeili Loop Evolution Plan v1

## Core Role of Loop in the New Framework

In the report-driven architecture, loop still matters, but its role changes.

It is no longer mainly:

- free hypothesis exploration
- one weak loop -> jump to a new unrelated direction

Instead, loop becomes:

- staged validation of one research thesis
- controlled evolution of proxy expressions
- gradual refinement from reconstruction to resonance

So the new meaning of loop is:

`reconstruct thesis -> tune parameters -> add resonance/filter -> only then revise hypothesis if needed`

## Core Principle

1. Reconstruct first.
2. Evolve expressions before evolving thesis.
3. Only revise the hypothesis after repeated evidence shows that the current mapping is weak or structurally wrong.

## Loop-by-Loop Plan

### Loop 1: Base Reconstruction

Goal:

- verify whether the report logic is observable in the current crypto dataset

Focus:

- simple price-extension proxy
- simple momentum-deceleration proxy
- simple overextension proxy
- simple non-confirmation proxy

Recommended factors:

1. `RANK(TS_PCTCHANGE($close, 6))`
2. `RANK(TS_PCTCHANGE($close, 6) - TS_PCTCHANGE($close, 24))`
3. `RANK(($close / ($vwap + 1e-8)) - 1)`
4. `RANK(TS_PCTCHANGE($close, 6)) - RANK($volume_change_24h)`

Do not do in Loop 1:

- complex thresholds
- multi-layer filters
- heavy multi-variable combinations
- thesis rewrite

### Loop 2: Parameter Evolution

Goal:

- keep the same thesis, but test whether the proxy is better expressed with different windows or scaling

Focus:

- window changes
- normalization changes
- short vs medium horizon tuning

Examples:

- `6 -> 3 / 9 / 12`
- `24 -> 18 / 30`
- rank version vs non-rank version
- ratio vs difference form

Do not do in Loop 2:

- introduce unrelated variable families
- rewrite the original report thesis

### Loop 3: Resonance and Filter Evolution

Goal:

- test in which local market states the report logic becomes stronger

Focus:

- spread stress filter
- VWAP-relative state filter
- volume confirmation or non-confirmation
- simple interaction terms

Examples:

- price extension × high spread
- price extension conditioned on weak volume confirmation
- high VWAP deviation + momentum weakening

This is where the system begins to answer:

- when does divergence matter most?
- what confirmation improves it?

### Loop 4: Controlled Hypothesis Refinement

Goal:

- refine the thesis only if needed after observing results from the first three loops

Possible outcomes:

- keep original thesis and continue deeper tuning
- narrow the thesis to a regime-specific version
- flip direction if evidence consistently suggests the opposite sign
- conclude that the report logic is weak in crypto under the current data constraints

Examples:

- from "top divergence leads to reversal" to
  "top divergence mainly works under high spread and weak volume confirmation"

- from "generic divergence" to
  "late-trend divergence after 24h extension"

## Why This Loop Structure Is Better

- It prevents the system from drifting too early.
- It preserves the original report thesis long enough to test it fairly.
- It separates:
  - thesis validity
  - proxy quality
  - parameter choice
  - regime dependence

This is much closer to real research workflow.

## Practical Interpretation

In this new setup:

- Loop 1 asks: "Can I reconstruct the report logic?"
- Loop 2 asks: "What parameterization expresses it best?"
- Loop 3 asks: "When is it strongest?"
- Loop 4 asks: "Do I still believe the original thesis?"

## Final Summary

Loop is still an evolution engine, but no longer a free-search engine.

Its new purpose is:

- preserve thesis continuity
- refine the expression layer first
- delay hypothesis mutation until the reconstruction evidence is sufficient
