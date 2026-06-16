# Changes Since GitHub Version `88ea893`

This document summarizes the main changes made after the previously pushed GitHub version `88ea893`.

## 1. Research-Report-Driven Input Flow

We changed the workflow from plain direction-only prompting to a report-grounded research flow.

Main additions:

- `main.py`
  - Added support for `--research-report-pdf`
  - Latest run can save logs, summary, and markdown report automatically
- `src/trading_agents/infra/pdf_reader.py`
  - Extracts raw text from PDF research reports
- `src/trading_agents/infra/research_report_loader.py`
  - Loads research report metadata, cleaned text, dataset feature list, and validation target into shared context
- `src/trading_agents/infra/report_cleaner.py`
  - Cleans disclaimers, legal text, contact blocks, and OCR/vertical-noise fragments before the first agent sees the report

Behavioral impact:

- The first agent now consumes actual PDF content instead of only a generic direction string
- Research report text is adapted into crypto research context before factor design starts

## 2. First Agent: From Free Hypothesis Generation to Structured Research Interpretation

We upgraded the hypothesis agent from a loose hypothesis generator into a structured research interpreter.

Main files:

- `src/trading_agents/agents/hypothesis/hypothesis_agent.py`
- `src/trading_agents/prompts/hypothesis_agent.json`
- `src/trading_agents/workflows/context_policies.py`

New structured outputs include:

- `thesis`
- `mechanism`
- `expected_direction`
- `market_structure_dependencies`
- `crypto_adaptation_summary`
- `proxy_reconstruction_plan`
- `hypothesis_evolution_state`

Behavioral impact:

- A-share specific mechanisms are now identified and rewritten into crypto-compatible logic
- `proxy_reconstruction_plan.input_features` is constrained to real parquet columns only
- `fwd_*` fields are reserved for labels and excluded from factor inputs

## 3. Second Agent: Report Replication Instead of Free Drift

We changed the experiment designer so the first report-driven loop behaves like a replication experiment rather than unconstrained search.

Main files:

- `src/trading_agents/agents/experiment_designer_agent.py`
- `src/trading_agents/prompts/experiment_designer_agent.yaml`

New behavior:

- Detects research stage from shared context
- Supports explicit stages:
  - `base_reconstruction`
  - `parameter_tuning`
  - `resonance_filtering`
  - `thesis_revision`
- First loop is constrained to simple report-grounded proxy reconstruction
- Later loops stay anchored to the same proxy family unless evidence clearly requires revision
- Factor count remains controllable from env

Behavioral impact:

- Reduced free variable-family drift
- Loop 1 now resembles research replication
- Loop 2+ can do controlled parameter tuning and lightweight confirmation logic

## 4. Stage-Driven Loop Evolution

We changed loop semantics from implicit loop-number guessing to explicit state-driven evolution.

Main files:

- `src/trading_agents/agents/feedback/feedback_summarizer_agent.py`
- `src/trading_agents/agents/experiment_designer_agent.py`
- `src/trading_agents/agents/hypothesis/hypothesis_agent.py`
- `src/trading_agents/workflows/context_policies.py`

New shared state:

- `research_stage`
- `research_stage_reason`

Behavioral impact:

- The system no longer assumes "loop 2 means X" by hardcoded index alone
- Feedback now chooses the next stage based on actual results
- The system can stay in `parameter_tuning` when the thesis is not yet exhausted

## 5. Feedback Semantics and Thesis-Revision Boundaries

We improved the semantic meaning of feedback without changing the overall downstream framework.

Main files:

- `src/trading_agents/agents/feedback/feedback_summarizer_agent.py`
- `src/trading_agents/prompts/feedback_summarizer_agent.json`
- `src/trading_agents/prompts/hypothesis_agent.json`

New semantic diagnosis categories:

- `thesis_invalid`
- `proxy_weak`
- `parameter_not_tuned`
- `implementation_unstable`
- `direction_mismatch`

Additional improvements:

- `failed` or `partial` backtests no longer qualify as replacing the best result
- Stage transition now references semantic diagnosis
- `thesis_revision` is now constrained so weak/noisy evidence does not flip a reversal thesis into a continuation thesis

Behavioral impact:

- Easier to distinguish thesis failure from weak proxy construction
- More conservative and interpretable hypothesis revision

## 6. Hypothesis JSON Parsing and Robustness

We strengthened the hypothesis-agent output parsing path significantly.

Main files:

- `src/trading_agents/infra/llm/parser.py`
- `src/trading_agents/agents/hypothesis/hypothesis_agent.py`

Improvements:

- Better stripping of `<think>` blocks
- Better extraction from fenced JSON/code blocks
- Control-character cleanup
- Trailing-comma repair
- Invalid-escape repair
- Unquoted-key repair
- Heuristic repair for unescaped quotes inside string values
- Final `ast.literal_eval` fallback
- Parse-failure retries at the hypothesis-agent layer

Behavioral impact:

- Greatly reduced fragility from dirty model outputs
- Eliminated previously observed `Hypothesis agent response parse failed` runs in later tests

## 7. LLM Error Observability and Retry Diagnostics

We improved visibility into provider-side and network-side failures.

Main files:

- `src/trading_agents/infra/llm/retry.py`
- `src/trading_agents/infra/llm/gateway.py`
- `src/trading_agents/infra/llm/providers/openai_compatible.py`
- `src/trading_agents/infra/llm/providers/azure_openai.py`

New observability:

- Retry scheduling events are now logged
- Structured error details now include fields like:
  - exception type
  - status code
  - request id
  - provider/body preview
  - cause type/message

Behavioral impact:

- Future `Connection error` cases can be distinguished from rate-limit, 5xx, or upstream overload cases more clearly

## 8. Data-Schema Alignment

We aligned factor-generation inputs to the actual parquet schema and fixed evaluation target semantics.

Main files touched across the codebase:

- hypothesis / experiment prompts
- workflow shared context
- evaluation path

Behavioral impact:

- Factor inputs are now restricted to real non-`fwd_*` feature columns
- Validation target remains `fwd_returns_24h`

## 9. Reporting and Draft Artifacts

We added report-driven design documentation and draft artifacts for the research-report workflow.

Main directories/files:

- `docs/research_report_input_architecture.md`
- `docs/research_report_pdf_phase1_plan.md`
- `docs/research_report_drafts/`

These documents capture:

- report-to-hypothesis architecture
- staged rollout plan
- draft hypothesis packages and experiment specs
- loop evolution plans

## 10. Retained Run Artifacts

Per the current cleanup request, only the latest run artifacts are retained:

- latest log directory:
  - `logs/alpha_factor_mining_loop/20260423_140828`
- latest markdown report:
  - `reports/factor_mining_report_20260423_140828_20260423_145702.md`

Older generated log/report artifacts were removed.

## Overall Summary

Relative to `88ea893`, the project has shifted from:

- mostly generic factor mining with prompt-driven iteration

to:

- research-report-grounded factor mining with:
  - PDF ingestion
  - report cleaning
  - structured thesis extraction
  - crypto adaptation
  - stage-driven loop evolution
  - stronger parsing and error observability
  - more explicit semantic feedback

Short version:

- The workflow is now more stable, more interpretable, and more aligned with report-driven research methodology, even though raw factor performance may still require further iteration.
