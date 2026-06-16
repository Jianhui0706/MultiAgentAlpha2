# Research Report Driven Factor Mining

## Core Idea

The target architecture is:

`research report / discretionary view -> structured information extraction -> crypto market adaptation -> factor reconstruction -> parameter evolution -> resonance / filter evolution -> existing evaluation framework`

This means we do **not** change the final evaluation logic first.

The main change is to improve the **upstream input and hypothesis formation** process, so the system starts from:

- report thesis
- market mechanism
- regime assumptions
- invalidation conditions
- crypto-specific proxy mapping

instead of starting from a generic free-form initial direction.

## Is This Summary Correct?

Yes, the high-level summary is correct:

1. Input a research report or discretionary trading view.
2. Extract the key information.
3. Adapt the original market logic to crypto market structure.
4. Then continue with a factor-mining workflow similar to the current framework.

But there is one important refinement:

It is not just:

`extract -> adapt -> same framework`

It is more precisely:

`extract thesis -> translate mechanism -> adapt to crypto -> reconstruct observable proxies using current dataset -> run the existing hypothesis / factor design / coding / backtest / feedback loop`

The crucial extra step is **proxy reconstruction**.

Because many report concepts are not directly present in the current dataset, they need to be mapped into available crypto features.

## New Workflow

### Stage 1: Research Input

Input may come from:

- factor report
- market behavior report
- discretionary trader note
- internal thesis from PM / boss / researcher

### Stage 2: Structured Extraction

The system should extract:

- `thesis`
- `mechanism`
- `expected_direction`
- `core_signals`
- `regime`
- `invalidation`
- `suggested_horizon`
- `market_structure_dependencies`

### Stage 3: Crypto Adaptation

This is the most important new step.

The system should determine:

- which parts of the original report logic are market-specific
- which parts are transferable to crypto
- what crypto-equivalent mechanism should replace the original one
- which parts should be dropped entirely

Examples:

- A-share opening gap -> crypto short-window price jump / re-pricing proxy
- turnover-rate logic -> volume / quote-volume proxies
- auction or close effect -> fixed UTC session effect or intraday slice effect

### Stage 4: Proxy Reconstruction

Translate adapted logic into variables that exist in the current parquet dataset.

Examples of current usable features:

- `open`, `high`, `low`, `close`
- `volume`, `volume_quote`
- `returns_1h`, `returns_4h`, `returns_12h`, `returns_24h`
- `log_return_1h`
- `volume_change_1h`, `volume_change_24h`
- `high_low_spread`
- `vwap`
- `open_close_return`

Labels remain:

- `fwd_returns_24h` for validation

### Stage 5: Factor Reconstruction

Before free evolution, the system should first reconstruct the report's core logic into a small number of base factors.

Priority:

1. single-variable proxy
2. direct two-variable interaction
3. simple normalized version

This is the "can the report logic be reproduced in our data?" stage.

### Stage 6: Controlled Evolution

Only after basic reconstruction should the system evolve:

1. parameter tuning
2. threshold tuning
3. normalization changes
4. resonance with related variables
5. regime filters
6. limited combinations

This is still factor mining, but now it is **report-anchored evolution**, not unconstrained exploration.

### Stage 7: Existing Evaluation

The downstream evaluation framework can remain unchanged for now:

- IC
- Rank IC
- ICIR
- layer returns
- long-short metrics
- sharpe
- drawdown

So the main architectural change is upstream, not downstream.

## Does This Mean We Only Modify The First Agent?

Short answer: **No.**

Long answer: **Phase 1 can mostly modify the first agent and input layer, but a complete design requires small coordinated changes in later agents too.**

## What Changes In Phase 1

Phase 1 can focus mainly on the entry layer and the first agent.

### Main new components

- research report input template
- structured extraction module
- crypto adaptation module
- hypothesis agent prompt/context upgrade

### Why the first agent is the main place to start

Because the largest gap today is not in evaluation.

The largest gap is that the system currently starts from a too-thin initial idea.

So the first practical upgrade is:

- give the first agent much richer structured input
- let it generate a crypto-adapted hypothesis instead of a generic free-form hypothesis

## Why Later Agents Still Need Some Changes

Even if the first agent becomes report-aware, later agents must also understand that the workflow is now thesis-driven.

### 1. Experiment designer still needs changes

It should no longer behave like fully free expression search.

It should:

- preserve the core thesis
- reconstruct observable proxies first
- separate:
  - core signal
  - confirmation signal
  - regime filter
  - invalidation filter
- evolve in a controlled order:
  - reconstruction
  - parameters
  - resonance
  - filters

So yes, `experiment_designer_agent` must change at least in prompt logic and output structure.

### 2. Feedback agent also needs changes

If the workflow is report-driven, feedback should not only say:

- metrics improved or not

It should also say:

- whether the original report thesis is supported
- whether the crypto adaptation seems valid
- whether the chosen proxy is too weak / too indirect
- whether the logic fails because the thesis is wrong or because the proxy is poor

So feedback semantics need to become more thesis-aware.

### 3. Report generator should eventually change

Final markdown output should evolve from a pure loop report into a research-style validation note:

- source thesis
- crypto adaptation
- proxy mapping
- factor reconstruction
- evolution path
- validation conclusion

## Best Practical Interpretation

So the right answer is:

- **Phase 1:** mostly modify the first agent and the input layer
- **Phase 2:** update experiment designer so it becomes reconstruction-first rather than free-search-first
- **Phase 3:** update feedback/report so conclusions are tied back to the original report thesis

## Recommended Implementation Order

1. Add a structured research-report input template.
2. Add a report extraction + crypto adaptation layer.
3. Feed the structured output into the current hypothesis agent.
4. Update experiment designer to generate base proxy factors first.
5. Update feedback to judge thesis validity vs proxy quality separately.
6. Update report generation to show the research-to-factor path.

## Final Conclusion

Your summary is directionally correct:

`input report -> extract information -> adapt to crypto -> then continue in the existing framework`

But the more exact version is:

`input report -> structured extraction -> crypto market adaptation -> dataset proxy reconstruction -> factor mining within the current framework`

And architecturally:

- it is **not only** the first agent
- but the **first agent is the right place to start**
- the rest of the framework can initially remain mostly intact
- then later agents should be adjusted to become thesis-aware rather than purely search-driven
