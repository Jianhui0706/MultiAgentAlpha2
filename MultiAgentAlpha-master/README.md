# multi-agent-core 运行说明

## Future Improvement Guideline: Multiple Research Reports As One Combined Hypothesis Input

Goal: support multiple input reports without rewriting the downstream multi-agent workflow. The preferred design is:

`multiple PDFs -> parse each report -> combine/synthesize into one research_report payload -> existing hypothesis_agent -> existing experiment loop`

This means downstream agents should still receive one compact `research_report`, but that report is a synthesized combination of several source reports.

### Code Areas To Revise

1. **CLI input in `main.py`**
   - Current entry point uses one `--research-report-pdf`.
   - Add a new multi-report argument such as `--research-report-pdfs report1.pdf report2.pdf report3.pdf`.
   - Keep the old single-report argument for backward compatibility.
   - Acceptance: both old single-report runs and new multi-report runs should work.

2. **Report payload construction in `src/trading_agents/workflows/alpha_factor_mining_workflow.py`**
   - Current methods such as `from_report_pdf(...)` and `run_report_driven_loop_session(...)` expect one PDF path.
   - Add a multi-report path that calls a new loader helper and writes one combined `research_report` object into shared payload.
   - Do not change downstream loop execution unless necessary.
   - Acceptance: `shared_context.payload["research_report"]` should exist with the same top-level shape expected by current agents.

3. **Report loader in `src/trading_agents/infra/research_report_loader.py`**
   - Current `build_research_report_payload(...)` parses one file.
   - Add a helper such as `build_combined_research_report_payload(report_paths, ...)`.
   - For each report, preserve metadata: file path, file name, page count, cleaned text length, and short extracted summary.
   - Combine all reports into one compact `report_text`, not raw full-text concatenation if reports are long.
   - Acceptance: combined payload should include `source_reports` plus a synthesized `report_text`.

4. **Synthesis format**
   - The combined `report_text` should be structured, for example:
     - report list and metadata
     - per-report thesis summary
     - common mechanisms
     - conflicting mechanisms
     - transferable market logic
     - non-transferable market assumptions
     - candidate proxy families
   - Do not simply append all PDF text forever. The combined input should be a controlled synthesis.
   - Acceptance: prompt length should remain bounded and visible in `llm_raw_io.jsonl`.

5. **Hypothesis agent in `src/trading_agents/agents/hypothesis/hypothesis_agent.py`**
   - Ideally no major architecture change is needed.
   - It already reads `research_report.report_text`, `research_report.source`, `dataset_features`, and `validation_target`.
   - Add support for optional `research_report.source_reports` in diagnostics and prompt metadata.
   - Acceptance: logs should show all source report names and the final chosen combined thesis.

6. **Prompt files in `src/trading_agents/prompts/hypothesis_agent.json`**
   - Adjust wording from “A research report PDF has been provided” to support “one or more research reports have been provided”.
   - Instruct the model to synthesize rather than average opinions blindly.
   - It should identify consensus, disagreement, and the final selected mechanism anchor.
   - Acceptance: hypothesis output should explain why the selected thesis was chosen from the combined reports.

7. **Report output in `scripts/generate_report.py`**
   - Add display of source report list when available.
   - Include the combined report thesis and any conflicts between reports.
   - Acceptance: final report should make it clear which PDFs contributed to the hypothesis.

8. **Tests / validation scripts**
   - Add a lightweight validation script that uses two or three small fake report texts.
   - It should not run qlib.
   - It should verify payload shape, source report metadata, prompt length, and hypothesis-agent compatibility.
   - Acceptance: validation can run quickly before a long backtest.

### Acceptance Criteria

The multi-report version should be accepted only if all of the following are true:

1. Single-report runs remain backward compatible with the current CLI.
2. Multi-report runs record every source filename in logs and final report.
3. `research_report.report_text` is a synthesized combination, not uncontrolled full-text concatenation.
4. `research_report.source_reports` contains per-report metadata.
5. `hypothesis_agent` receives one combined report context and produces one primary thesis.
6. The hypothesis output explains consensus, conflicts, selected mechanism, and proxy reconstruction plan.
7. `llm_raw_io.jsonl` shows prompt length remains controlled.
8. Continuity gate still tracks the selected mechanism anchor after loop 1.
9. Experiment designer does not need to know whether the original input was one report or multiple reports.
10. A fast validation script passes without running qlib.

## 2026-05-08 提交记录

本轮版本重点收口 `ExperimentDesignerAgent` 的上下文体积与首轮表达式合法率，减少无效重试，并把本次修改范围明确记录到项目文档中。

1. **ExperimentDesigner 批量语义校验**
   - 将 `_run_consistency_gate()` 从“逐因子单次 verifier”改为“整批因子一次 verifier”。
   - 同一批候选因子共用一次语义校验上下文，减少重复发送 hypothesis、evolution guidance 和 advisory records。
   - `experiment_spec.consistency_gate` 增加 `llm_call_count`，`quality_gate.mode` 更新为 `llm_batch_verifier_with_metric_advisory`，便于从日志直接确认调用模式。

2. **Research report 注入压缩**
   - `ExperimentDesignerAgent` 不再把 research report 的长文本或重复片段直接注入主生成 prompt。
   - 新增紧凑报告摘要，仅保留 `thesis / mechanism / caveats / proxy families / compact proxy reconstruction plan`。
   - `experiment_designer_agent.yaml` 改为统一使用 `report_context_summary`，减少 report grounding 的重复 prompt 文案。

3. **Retry 反馈压缩**
   - `duplication_feedback` 在 retry 阶段改为压缩注入，不再把完整长反馈无限叠加到下一轮 prompt。
   - 新增环境变量：
     - `EXPERIMENT_DESIGNER_FEEDBACK_ITEM_LIMIT`
     - `EXPERIMENT_DESIGNER_FEEDBACK_CHAR_LIMIT`
   - 默认仅保留最近少量高优先级反馈，用于表达式修复，不改变完整 rejection feedback 的日志留存。

4. **规则包加强：提升首轮合法表达式比例**
   - `SKILL.md`、`core_rules.md`、`output_contract.md` 增加更强的提交前自检要求：
     - feature 必须来自真实 schema
     - feature 必须采用 `$feature_name`
     - function 必须受支持且 arity 正确
     - `variables` 必须与 `expression` 对齐
     - 输出必须是严格 JSON 对象
   - 明确阻止常见非法输出：
     - `$volume_change_4h`
     - `$volume_change_12h`
     - `$return`
     - `$returns`
     - `TS_PERCENTILE`

5. **Prompt 规则继续收敛**
   - `experiment_designer_agent.yaml` 进一步减少与规则包重复的历史说明。
   - report 相关说明改为引用紧凑摘要，不再重复逐段注入 thesis / mechanism / proxy plan。

6. **本次验证**
   - `python -m py_compile src/trading_agents/agents/experiment_designer_agent.py` 通过。
   - `pytest -q tests/test_experiment_designer_prompt_context.py tests/test_experiment_designer_skill_loader.py tests/test_experiment_designer_validation_regression.py tests/test_feature_whitelist.py` 通过。
   - 新增/更新测试覆盖：
     - compact report summary 注入
     - batch verifier 仅调用一次
     - duplication feedback retry 压缩
     - rules 显式拦截高频非法 feature / function

## 2026-05-02 提交记录

本轮版本重点把“多研报合成输入”和“扣手续费后的净收益评价”接入主流程，并增强最终报告的可验收性。

1. **多研报输入链路**
   - `main.py` 支持新的 `--research-report-pdfs report1.pdf report2.pdf ...` 参数，同时保留原有 `--research-report-pdf` 单研报用法。
   - workflow 会把多个 PDF 合成为一个 `research_report` payload， downstream agents 仍然只消费一个统一研究报告对象。
   - `research_report.source_reports` 记录每份 PDF 的文件名、路径、页数和摘要信息，最终 report 也会展示来源列表。

2. **紧凑多研报 synthesis**
   - `research_report_loader` 不再简单拼接全部原文，而是生成受控合成文本。
   - 合成内容包含 per-report thesis、共识机制、冲突/保留意见、可迁移逻辑、不可迁移市场假设和 proxy candidates。
   - 新增快速校验脚本，用于不跑 qlib 的情况下检查 combined report payload 形状和兼容性。

3. **Hypothesis agent 多研报感知**
   - prompt 与 agent diagnostics 支持 “one or more research reports”。
   - Hypothesis 输出需要关注 consensus、conflict、selected mechanism anchor 和 source report names。
   - Continuity gate 后续仍围绕选定的单一 mechanism anchor 演化，避免多研报输入导致主题漂移。

4. **单边万4交易费接入回测**
   - `backtest_engine.yaml` 增加 `transaction_cost.one_way_fee_rate: 0.0004`，即 4 bps one-way transaction fee。
   - 本地 long-short 分层回测按 top/bottom layer 权重变化估算 `long_short_turnover`，并扣除交易费。
   - 指标中同时保留 `gross_sharpe`、`net_sharpe`、`transaction_cost_mean`、`long_short_turnover`，其中最终 `sharpe` 与 `net_sharpe` 对齐。
   - qlib baseline / combined config 的 `open_cost`、`close_cost` 统一为 `0.0004`，`min_cost` 改为 `0`。

5. **Top 6 净 Sharpe 分层收益图**
   - 最终 report 的 Top 图从 Top 3 扩展为 Top 6，并组合成 2 行 3 列总图。
   - 因子选择口径改为扣费后的 `net_sharpe` 从大到小，旧日志缺少 `net_sharpe` 时回退到 `sharpe`。
   - 图中累计收益曲线展示扣除单边万4后的净收益，标题显示 `Net Sharpe` 和 `fee=4.0bps/side`。
   - 每个子图标题第二行展示 qlib/factor expression，方便人工复核因子表达式。

6. **本次验证**
   - `python -m py_compile scripts/generate_report.py` 通过。
   - `python -m py_compile src/trading_agents/agents/backtest_runner_agent.py scripts/generate_report.py` 通过。
   - `git diff --check` 通过。
   - synthetic fee sanity check 显示扣费后 `net_sharpe` 低于 `gross_sharpe`，交易费已进入收益计算。

## 2026-04-28 版本 B 提交记录

本次版本 B 相比上一版，重点从“能跑通多轮机制锚定实验”继续推进到“日志可验收、表达式更稳、回测评价更可信、报告更直观”。主要区别如下：

1. **Continuity gate 与自然语言假设同步**
   - 当 `continuity_enforced=True` 时，不再只夹回 `mechanism_anchor` 和 `expected_direction`。
   - 同步夹回上一轮权威的 `hypothesis`、`thesis`、`mechanism` 文本，避免出现“结构字段说负向，文字假设说正向 continuation”的矛盾。
   - diagnostics 增加 `natural_language_enforced` 和 `natural_language_enforced_fields`，方便在 log 中直接确认是否发生硬性夹回。

2. **LLM raw I/O 日志修复**
   - `llm_raw_io.jsonl` 改成真正的一行一个 JSON，后续可以直接逐行解析。
   - Loop runner 写入真实 `loop_index`，不再全部显示为 0。
   - 每次 LLM 调用仍保留 system/user 输入长度、请求、响应、错误和 retry 信息，便于评估 token 成本与慢轮次来源。

3. **表达式函数签名与 JSON 容错**
   - 增加函数参数个数校验，提前拦截不合法表达式。
   - 自动修复常见模型误用：`MAX(A,n)`、`MIN(A,n)`、`MEDIAN(A,n)`、`MEAN(A,n)`、`STD(A,n)` 会改写为对应的 `TS_*` rolling 函数。
   - `factor_coder` 和 `experiment_designer` 都加入本地兜底，避免这类错误拖到 qlib 计算阶段才失败。
   - JSON parser 增加数组/对象闭合符错配修复，缓解模型输出里 `[` 被 `}` 错关导致的解析失败。

4. **Counterfactual 与阶段推进继续收口**
   - Counterfactual 保持“批判当前假设”的定位：使用相同 anchors，提出相反经济解释，而不是简单换变量或复制主假设。
   - 缺少 mainline/counterfactual 数量时不再用本地模板自动补齐，而是通过 retry 和 shortfall diagnostics 暴露问题。
   - 早期弱信号不再过快进入 thesis replacement，优先走 `parameter_tuning` 或 `proxy_reconstruction`。

5. **Sharpe 评分口径修复**
   - 回测中保留 `raw_sharpe`，同时根据 Rank IC 方向对多空组合做方向对齐后计算最终 `sharpe`。
   - 对反向有效因子，不再因为多空方向没翻转而把 Sharpe 误判为 0 或负向。
   - 回测报告记录 `direction_sign`、`direction_aligned_by_rank_ic`、`sharpe_sampling_step` 和 `sharpe_annualization_factor`，让 Sharpe 语义可追踪。

6. **分组收益曲线与报告增强**
   - 报告会按最终方向对齐后的 Sharpe 选出 Top 因子。
   - 对 Top 因子生成分组累计收益图，展示多个 signal group 与全市场等权 benchmark 的走势。
   - 报告增加分组收益、理论内/理论外对照、factor group shortfall、counterfactual thesis/mechanism/critique 等信息，便于人工验收。

7. **计算效率与缓存链路**
   - 回测阶段按唯一 expression 分组，重复表达式只计算一次，再复用到同组因子。
   - 因子 signal cache 信息继续向 report 和 factor library 传递。
   - 主流程结束后清理 `factor_library/cache`，减少长跑后本地缓存膨胀。

8. **本次快速验证**
   - `py_compile` 通过。
   - `MAX/MEDIAN` 函数误用修复回归测试通过。
   - JSON 闭合符错配修复测试通过。
   - `scripts/validate_counterfactual_anchor_binding.py` 通过。

## 2026-04-27 提交记录

## Gump 运行与遗传迭代模式说明

### 1) 运行命令

在项目根目录执行：

```powershell
cd "d:\ai powered crypto quant\quantaAlpha\QuantaAlpha-main\multi-agent-core"
python examples/run_alpha_factor_mining_loop.py
```

### 2) Loop 次数设置

`examples/run_alpha_factor_mining_loop.py` 中通过 `loop_count` 控制循环轮数：

```python
loop_count = 100
```

按需改成你希望的轮数后再运行。

### 3) 初始种子注入

首轮种子可以通过环境变量 `INITIAL_DIRECTION` 注入；在 evolution 模式下，它会作为初始化 Agent A0 生成多条初始假设 H0 的输入。

PowerShell 示例：

```powershell
$env:INITIAL_DIRECTION = "山寨币在过热时可能会发生反转，基于这个假设可以发掘盈利因子。"
python examples/run_alpha_factor_mining_loop.py
```

如果不设置，则脚本使用内置默认方向。

### 4) `.env` 文件说明（智谱）

脚本会读取项目根目录下的 `.env`（以及可选的 `.env.local`）。

最小示例：

```env
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=你的智谱APIKey
ZHIPU_MODEL=glm-4-flash
```

### 5) `main.py` 启动参数

正式入口也支持从项目根目录直接运行：

```powershell
python main.py --loop-count 1 --initial-direction "山寨币在过热时可能会发生反转，基于这个假设可以发掘盈利因子。"
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--panel-data-path` | 面板数据路径，默认读取 `data/panel_data.parquet`。 |
| `--loop-count` | 普通循环模式的轮数。 |
| `--initial-direction` | 初始研究方向。 |
| `--initial-hypothesis` | 初始假设文本；设置后优先于 `--initial-direction` 作为首轮种子。 |
| `--research-report-pdf` | 可选研究报告 PDF，用于给首轮 agent 注入报告上下文。 |
| `--stop-on-error` | 普通循环模式下遇到错误立即停止。 |
| `--retry-per-round` | 普通循环模式下每轮失败后的重试次数。 |
| `--debug` | 输出更详细的调试事件。 |
| `--no-report-pdf` | 只生成 Markdown 报告，跳过 PDF 转换。 |
| `--train-sharpe-threshold` | 可选覆盖训练期 Sharpe 过滤阈值；不传时使用 `backtest_engine.yaml` 中的 `sample_windows.train_filter.min_value`。 |

### 6) 样本内 / 样本外 Sharpe 过滤

回测配置位于 `src/trading_agents/configs/qlib/backtest_engine.yaml`。当前默认样本窗口为：

- 样本内 train：`2021-01-01 <= datetime < 2025-06-01`
- 样本外 test：`datetime >= 2025-06-01`
- 默认过滤条件：训练期 `sharpe > 2.0`

训练和挖掘阶段只使用 train window；test window 不进入训练过程。每轮回测结束后，`FactorLibraryManager` 只保存训练期 Sharpe 通过阈值的因子。整个 session 结束后，`main.py` 会统一对本次 run 保存的候选因子跑一次 holdout test，并在 run 目录输出：

```text
logs/alpha_factor_mining_loop/<run_id>/holdout_validation.json
```

该文件会汇总每个候选因子的 `train_sharpe`、`test_sharpe`、`passed_train_threshold` 和 `passed_test_threshold`。`summary.json` 同时记录 `holdout_validation_file` 与 `holdout_validation` 摘要。

可通过 CLI 临时覆盖训练期 Sharpe 阈值：

```powershell
python main.py --train-sharpe-threshold 0.5 --loop-count 1
```

`bin/debug.sh` 为了更快看到候选因子，默认传入 `--train-sharpe-threshold`；正式长跑不传该参数时，仍使用配置文件中的默认阈值。

### 7) 遗传迭代模式参数

开启遗传迭代模式后，入口会运行：

```text
original -> mutation -> crossover -> mutation -> crossover -> ...
```

PowerShell 示例：

```powershell
python main.py `
  --evolution-mode `
  --evolution-max-rounds 3 `
  --evolution-num-initial-hypotheses 5 `
  --evolution-mutation-tasks-per-round 3 `
  --evolution-crossover-tasks-per-round 2 `
  --evolution-crossover-group-size 2 `
  --evolution-crossover-n-groups 1
```

遗传迭代参数：

| 参数 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--evolution-mode` | `EVOLUTION_MODE` | `false` | 开启遗传迭代入口；不传时保持普通循环模式。 |
| `--evolution-max-rounds` | `EVOLUTION_MAX_ROUNDS` | `6` | 遗传迭代最大轮数。 |
| `--evolution-num-initial-hypotheses` | `EVOLUTION_NUM_INITIAL_HYPOTHESES` | `5` | 初始化 Agent A0 在第 0 轮直接生成的初始假设 H0 数量；旧参数 `--evolution-num-initial-directions` / `EVOLUTION_NUM_INITIAL_DIRECTIONS` 仍兼容。 |
| `--evolution-mutation-tasks-per-round` | `EVOLUTION_MUTATION_TASKS_PER_ROUND` | `3` | EvolutionAgent 每轮规划的 mutation 任务数。 |
| `--evolution-crossover-tasks-per-round` | `EVOLUTION_CROSSOVER_TASKS_PER_ROUND` | `2` | EvolutionAgent 每轮规划的 crossover 任务数。 |
| `--evolution-evolved-directions-per-round` | `EVOLUTION_EVOLVED_DIRECTIONS_PER_ROUND` | `5` | 兼容旧 joint-seed 模式；当前默认调度直接执行 mutation/crossover 任务，不再用它压缩出单一下一代方向。 |
| `--evolution-mutation-parent-top-n` | `EVOLUTION_MUTATION_PARENT_TOP_N` | `10` | 提供给 EvolutionAgent 的 mutation 候选父轨迹数量。 |
| `--evolution-crossover-parent-top-n` | `EVOLUTION_CROSSOVER_PARENT_TOP_N` | `10` | 提供给 EvolutionAgent 的 recent/crossover 候选轨迹数量。 |
| `--evolution-mutation-top-n` | `EVOLUTION_MUTATION_TOP_N` | `3` | 旧参数；作为规则 fallback 的 mutation 父轨迹数量。 |
| `--evolution-crossover-group-size` | `EVOLUTION_CROSSOVER_GROUP_SIZE` | `2` | 每个 crossover 组合包含的父轨迹数量。 |
| `--evolution-crossover-n-groups` | `EVOLUTION_CROSSOVER_N_GROUPS` | `2` | 旧参数；作为规则 fallback 的 crossover 组合数量。 |

也可以在 `.env` 或 `.env.local` 中配置：

```env
EVOLUTION_MODE=true
EVOLUTION_MAX_ROUNDS=3
EVOLUTION_NUM_INITIAL_HYPOTHESES=5
EVOLUTION_MUTATION_TASKS_PER_ROUND=3
EVOLUTION_CROSSOVER_TASKS_PER_ROUND=2
EVOLUTION_MUTATION_PARENT_TOP_N=10
EVOLUTION_CROSSOVER_PARENT_TOP_N=10
EVOLUTION_MUTATION_TOP_N=2
EVOLUTION_CROSSOVER_GROUP_SIZE=2
EVOLUTION_CROSSOVER_N_GROUPS=1
```

运行结束后，`logs/alpha_factor_mining_loop/<run_id>/summary.json` 会包含 `evolution_mode`、`trajectory_pool_stats`、`best_trajectories`、`rounds` 和 `lineage_tree`，用于查看每轮阶段和父子轨迹关系。

`bin/debug.sh` 当前设置为 `--evolution-max-rounds 1`、`--evolution-num-initial-hypotheses 2`。更长的 evolution 脚本会在第 0 轮先跑 2 条由初始化 Agent A0 生成的初始假设，之后按 QuantaAlpha profile 交替执行 mutation phase / crossover phase；若某轮所有任务失败，evolution runner 会提前结束。

## 2026-05-05 Gump Evolution 合并记录

本次将 `test/merge-gump-into-master` 合并到 `master`，把 Gump 分支的遗传进化工作流与当前主线的多研报、扣费回测和报告能力整合到一起。

1. **遗传进化工作流**
   - 新增 `DirectionDecomposerAgent` 和 `EvolutionAgent`，现在用于从初始种子生成多条初始假设 H0，并在后续轮次中规划 mutation / crossover 子假设。
   - 新增 `evolution_runner.py` 和 `evolution_guidance.py`，记录 `trajectory_id`、`parent_ids`、`phase`、`round_idx`、`lineage_tree` 和 `best_trajectories`。
   - `main.py` 新增 `--evolution-mode` 以及相关 evolution 参数，可通过 CLI 或 `.env` 控制遗传迭代轮数、初始假设数量和每轮 mutation/crossover 任务数。

2. **多研报输入串到 evolution**
   - `--research-report-pdf` / `--research-report-pdfs` 在 `--evolution-mode` 下不再被跳过。
   - 主流程会先解析单研报或多研报，生成 `research_report` payload，再把该 payload 交给 evolution workflow。
   - `bin/debug.sh` 提供多研报 + evolution 的快速 smoke test，并默认使用 `/opt/anaconda3/bin/python`，避免 `uv` 在 macOS 12 arm64 上解析 `onnxruntime` wheel 失败。

3. **回测与因子计算增强**
   - 新增 `factor_functions_matrix.py`，优先尝试矩阵化计算因子表达式，失败时回退到原 pandas/groupby 版本。
   - `backtest_runner_agent.py` 继续作为回测主接口，调用 `QlibFactorBacktestEngine.evaluate_with_qlib()`，并写回 `backtest_report`、`metrics`、`per_factor_metrics`。
   - 当前回测仍是本地 panel 分层回测为主，使用 qlib 初始化和 IC 工具；Sharpe、turnover、手续费和分层收益由项目代码计算。

4. **工程与测试**
   - 新增 evolution、direction decomposer、factor library、feature whitelist、backtest metrics 等测试文件。
   - `.gitignore` 增加对运行生成目录的忽略，`artifacts/trajectory_pool.json` 和旧 factor library 快照不再作为主线跟踪对象。
   - 合并前保留了当前主线的 `factor_library/all_factors_library.json`，没有采用 Gump 对运行生成 factor library 的删除替换。

5. **本次快速验证**
   - `bash -n bin/debug.sh` 通过。
   - `python -m py_compile main.py` 通过。
   - 关键 evolution、backtest、experiment designer、factor runtime 文件的 `py_compile` 通过。

6. **本次 merge 范围**
   - 合并来源分支：`test/merge-gump-into-master`。
   - 合并目标分支：`master`。
   - 关键提交链：`a79f015 Merge gump evolution workflow into master`、`08ecbd4 Wire report inputs into evolution debug workflow`、`4e8b68c Merge latest gump into test branch`、`0effa87 Fix test_feature_whitelist after master adds counterfactual group requirement`。
   - 主要新增文件：`bin/debug.sh`、`bin/debug.ps1`、`run.ps1`、`src/trading_agents/agents/direction_decomposer_agent.py`、`src/trading_agents/agents/evolution_agent.py`、`src/trading_agents/workflows/evolution_runner.py`、`src/trading_agents/workflows/evolution_guidance.py`、`src/trading_agents/factor_runtime/factor_functions_matrix.py`、`scripts/plot_group_returns.py`、`scripts/extract_selected_factors.py`、`tests/test_evolution_agent.py`、`tests/test_evolution_runner_planning.py`、`tests/test_direction_decomposer_agent.py`。
   - 主要修改文件：`main.py`、`scripts/generate_report.py`、`src/trading_agents/agents/backtest_runner_agent.py`、`src/trading_agents/agents/experiment_designer_agent.py`、`src/trading_agents/agents/factor_coder_agent.py`、`src/trading_agents/agents/hypothesis/hypothesis_agent.py`、`src/trading_agents/schemas/models.py`、`src/trading_agents/factor_runtime/factor_functions.py`、`src/trading_agents/prompts/experiment_designer_agent.yaml`、`src/trading_agents/prompts/hypothesis_agent.json`。
   - 主要移出跟踪：`artifacts/trajectory_pool.json`、`factor_library/all_factors_library_0425.json` 和若干旧示例报告文件。运行生成的 trajectory pool、factor library 快照和历史报告不再作为本次主线核心代码资产维护。

## 2026-05-06 Report Net Sharpe Log-Scale 图表改进

本次改进聚焦最终报告里的 Top 6 分层收益图，使图表口径与当前扣费回测逻辑保持一致，并提升极端累计收益场景下的可读性。

1. **Top 6 选图口径**
   - 报告继续按扣除单边万4手续费后的 `net_sharpe` 选择 Top 6 因子。
   - 旧日志缺少 `net_sharpe` 时回退到 `sharpe`，保留兼容性。
   - 报告中仍展示 `gross_sharpe`，方便对比手续费前后的表现差异。

2. **分层收益图展示**
   - Top 6 的单图和 2x3 合并图统一展示扣费后的净累计收益曲线。
   - y 轴改为 log scale，避免少数分组涨到几十倍、上百倍时把其他分组压扁。
   - 非正数、`NaN`、`Inf` 曲线值会在绘图阶段做安全处理，防止 log 坐标无法渲染。

3. **验证结果**
   - `python -m py_compile scripts/generate_report.py` 通过。
   - `git diff --check scripts/generate_report.py` 通过。
   - 已用 `logs/alpha_factor_mining_loop/20260505_180034` 重新生成报告，输出包含 6 张单因子分层图和 1 张 `top_net_sharpe_2x3_combined.png` 合并图。
