# Repository Notes

## Entry Points
- Primary entrypoint is `main.py`, not `examples/`. It auto-loads `.env` then `.env.local`, injects both repo root and `src/` into `sys.path`, and writes each run under `logs/alpha_factor_mining_loop/<run_id>/`.
- Useful run outputs are `progress.jsonl`, `llm_raw_io.jsonl`, `summary.json`, and per-loop `loop_###.json` files in that run directory.
- `run.ps1` is the simplest normal run wrapper; `bin/debug.ps1` and `bin/debug.sh` are the smallest checked-in evolution smoke-test entrypoints.

## Commands
- Use `uv run` for repeatable local commands. Verified examples:
- `uv run python -m pytest tests/test_backtest_runner_metrics.py`
- `uv run python scripts/validate_multi_report_payload.py`
- `uv run python -m py_compile main.py scripts/generate_report.py`
- Pytest uses the repo root `pyproject.toml` as its config root; `tests/conftest.py` adds `src/` to `sys.path`.
- There is no repo-local lint/typecheck config at root. For small changes, prefer targeted `pytest`, `py_compile`, and the relevant `scripts/validate_*.py` script over inventing a lint step.

## Architecture
- The real workflow wiring is in `src/trading_agents/workflows/alpha_factor_mining_workflow.py`.
- Normal loop agent order is: `HypothesisAgentV2` -> `ExperimentDesignerAgent` -> `FactorCoderAgent` -> `BacktestRunnerAgent` -> `FeedbackSummarizerAgent`.
- `main.py --evolution-mode` switches to `EvolutionRunner`; the important persisted state is `artifacts/trajectory_pool.json` plus the run summary lineage fields.
- The domain adapter is `CryptoCrossSectionDomainAdapter`; default data input is `data/panel_data.parquet`.

## Research Report Flow
- Single-report and multi-report inputs both end up as one `research_report` payload consumed downstream.
- Multi-report handling lives in `src/trading_agents/infra/research_report_loader.py` and `AlphaFactorMiningWorkflow.build_multi_report_driven_initial_payload(...)`.
- Do not replace the compact synthesis with raw PDF concatenation: the verified contract is bounded `report_text` plus `source_reports` metadata and `multi_report_synthesis_status`.
- The focused regression for this path is `uv run python scripts/validate_multi_report_payload.py`.

## Backtest Semantics
- Backtest configuration lives in `src/trading_agents/configs/qlib/backtest_engine.yaml`.
- Transaction cost is part of the expected behavior: `metrics.transaction_cost.one_way_fee_rate` is `0.0004` (4 bps one-way).
- `BacktestRunnerAgent` reports top-level `metrics` as the best single factor result, not a combined multi-factor portfolio; keep that distinction intact in code and docs.
- Signal cache files are written under `factor_library/cache`, but `main.py` also clears that cache at exit.

## Reporting
- `scripts/generate_report.py` is the report generator used by `main.py` after every run.
- Prefer `--no-report-pdf` unless the task is specifically about PDF output. PDF conversion depends on external tools/libraries (`pandoc`, `weasyprint`, or `reportlab`) and is intentionally optional.

## Generated Files
- `logs/`, `log/`, `reports/`, `artifacts/`, `factor_library/`, `results/`, `docs/`, and `data/` are gitignored runtime/data areas in this repo.
- Avoid editing files under those directories unless the task is explicitly about fixtures, generated outputs, or artifact inspection.
