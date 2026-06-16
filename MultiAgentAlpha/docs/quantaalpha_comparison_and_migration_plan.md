# QuantaAlpha vs Multi-agent-factor-mining

## Core Conclusion

QuantaAlpha's core idea is not "micro-refine one hypothesis within a single loop many times". Instead, it treats each completed loop result as a `trajectory`, stores it in a `trajectory pool`, and then performs trajectory-level evolutionary search:

`original exploration -> orthogonal mutation -> strategy crossover -> mutation -> crossover -> ...`

Our current project is still mainly a single-line refinement workflow:

`hypothesis -> experiment design -> code/backtest -> feedback -> next hypothesis`

This means QuantaAlpha is structurally better at preserving diversity, lineage, and branch-level search.

## Structural Comparison

| Dimension | QuantaAlpha | Current Project |
|---|---|---|
| Search unit | `StrategyTrajectory` | current loop shared context |
| Search shape | multi-branch evolutionary search | mostly single-chain iterative refinement |
| Memory | trajectory pool with lineage | loop history and best-result memory |
| Mutation | explicit orthogonal mutation operator | implicit prompt-based refinement |
| Crossover | explicit multi-parent hybrid operator | none |
| Parent selection | RankIC-driven + diversity-aware | no branch parent selection |
| Branch persistence | yes | no explicit branch abstraction |
| Optimization pressure | trajectory-level `RankIC` | per-loop metrics and feedback |
| Evolution state | phase, round, parent ids, direction ids | hypothesis evolution state only |

## What QuantaAlpha Does Well

### 1. Multi-direction initialization

QuantaAlpha first expands an initial research direction into multiple parallel `direction`s, then runs an `original` loop for each direction.

This gives it broad initial coverage before exploitation starts.

Relevant files:

- [experiment.yaml](/Users/zhangzekun/Desktop/QuantaAlpha-main/configs/experiment.yaml)
- [factor_mining.py](/Users/zhangzekun/Desktop/QuantaAlpha-main/quantaalpha/pipeline/factor_mining.py)
- [controller.py](/Users/zhangzekun/Desktop/QuantaAlpha-main/quantaalpha/pipeline/evolution/controller.py)

### 2. Trajectory as the persistent search object

Each completed loop is turned into a `StrategyTrajectory` with:

- hypothesis
- factor expressions
- factor code
- backtest metrics
- feedback
- phase
- round index
- parent ids

This is much richer than only remembering "latest hypothesis" or "best metrics".

Relevant file:

- [trajectory.py](/Users/zhangzekun/Desktop/QuantaAlpha-main/quantaalpha/pipeline/evolution/trajectory.py)

### 3. Mutation is explicit and orthogonal

Mutation is not parameter tuning. It is single-parent, orthogonal exploration:

- one parent trajectory at a time
- use parent hypothesis, factors, metrics, feedback
- ask LLM to propose a strategy with different data dimensions and logic

Orthogonality is prompt-defined rather than numerically verified.

Relevant files:

- [mutation.py](/Users/zhangzekun/Desktop/QuantaAlpha-main/quantaalpha/pipeline/evolution/mutation.py)
- [evolution_prompts.yaml](/Users/zhangzekun/Desktop/QuantaAlpha-main/quantaalpha/pipeline/prompts/evolution_prompts.yaml)

### 4. Crossover is explicit strategy fusion

Crossover is multi-parent hybridization, not expression concatenation.

It reads each parent trajectory's:

- hypothesis
- factors
- metrics
- feedback

Then it asks the LLM to produce:

- `hybrid_hypothesis`
- `fusion_logic`
- `innovation_points`
- `expected_benefits`

That hybrid guidance is then fed into the next hypothesis generation round.

Relevant files:

- [crossover.py](/Users/zhangzekun/Desktop/QuantaAlpha-main/quantaalpha/pipeline/evolution/crossover.py)
- [experiment.yaml](/Users/zhangzekun/Desktop/QuantaAlpha-main/configs/experiment.yaml)

### 5. Parent selection is not fully random

QuantaAlpha supports:

- `best`
- `random`
- `weighted`
- `weighted_inverse`
- `top_percent_plus_random`

By default it leans toward exploitation with `parent_selection_strategy: best`, but still encourages diversity with `prefer_diverse_crossover: true`.

This is a good balance between:

- exploiting high RankIC branches
- preserving cross-direction diversity

## What It Is Actually Optimizing

At the trajectory level, QuantaAlpha mainly optimizes `RankIC`.

`StrategyTrajectory.get_primary_metric()` returns `RankIC`, and later parent selection and ranking are driven by it.

That means its evolutionary pressure is:

- high `RankIC` trajectories are more likely to become parents
- mutation introduces orthogonal exploration
- crossover adds diversity-aware recombination

Relevant file:

- [trajectory.py](/Users/zhangzekun/Desktop/QuantaAlpha-main/quantaalpha/pipeline/evolution/trajectory.py)

## End-to-End Evolution Flow

1. Start from one initial research direction.
2. If planning is enabled, expand it into multiple parallel directions.
3. Run one full loop for each direction and produce `original trajectories`.
4. Save every loop result into `trajectory pool`.
5. Enter mutation round:
   generate orthogonal guidance from each selected parent trajectory, then run full loops again.
6. Enter crossover round:
   select parent groups by `RankIC + diversity`, generate hybrid guidance, then run full loops again.
7. If `max_rounds` is not reached:
   use the latest crossover outputs as the main mutation inputs and continue `mutation -> crossover`.
8. Rank final trajectories by `RankIC` and output top branches with lineage.

## Why This Design Is Strong

- It is better at maintaining factor diversity than a single-chain refinement loop.
- Mutation helps jump away from local optima.
- Crossover helps combine useful ideas from different branches.
- Lineage makes post-analysis much easier.
- It separates "explore a new direction" from "fuse two directions", which is cleaner than overloading one prompt.

## Recommended Migration Targets For This Repo

The best concepts to import into our current project are:

1. `trajectory pool`
2. `mutation operator`
3. `crossover operator`
4. `evolution controller`

If we add these four abstractions, our system can move from single-line hypothesis refinement to multi-branch evolutionary search.

## Migration Mapping

| QuantaAlpha concept | Suggested mapping in this repo |
|---|---|
| `StrategyTrajectory` | `FactorTrajectory` object wrapping one loop result |
| `TrajectoryPool` | persisted pool under `artifacts/trajectory_pool.json` |
| `MutationOperator` | prompt-driven branch mutation module |
| `CrossoverOperator` | hybrid hypothesis generator from multiple branches |
| `EvolutionController` | next-task scheduler above `AlphaFactorMiningWorkflow` |
| `RoundPhase` | `original / mutation / crossover` enum |

## Proposed Migration Plan

### Phase 1: Trajectory foundation

Goal:

- keep current workflow unchanged
- add branch-level persistence first

Tasks:

- define `FactorTrajectory` dataclass
- persist one trajectory per completed loop
- save:
  - hypothesis
  - hypothesis_structured
  - experiment_spec
  - factor list
  - metrics
  - per_factor_metrics
  - feedback
  - phase
  - round_idx
  - trajectory_id
  - parent_ids
- write and load from `artifacts/trajectory_pool.json`

Expected impact:

- no behavior break
- immediate branch lineage visibility

### Phase 2: Controller skeleton

Goal:

- move from "for loop over loop_count" to "controller decides next task"

Tasks:

- introduce `EvolutionPhase` enum
- add controller state:
  - current round
  - current phase
  - active branch ids
  - mutation targets
  - crossover groups
- keep first version simple:
  - round 0 = original
  - round 1 = mutation
  - round 2 = crossover

Expected impact:

- evolution becomes explicit instead of implicit

### Phase 3: Mutation operator

Goal:

- stop treating every next hypothesis as a generic refinement
- separate orthogonal exploration from local refinement

Tasks:

- add `MutationOperator`
- input:
  - parent trajectory
  - feedback
  - metrics
  - factor summary
- output:
  - mutated hypothesis guidance
  - orthogonality rationale
  - expected signal dimension
- feed this guidance into `hypothesis_agent`

Recommended rule:

- first mutation round mutates each original trajectory
- later mutation rounds mutate the latest crossover trajectories

### Phase 4: Crossover operator

Goal:

- allow branch recombination

Tasks:

- add `CrossoverOperator`
- select parent groups from trajectory pool
- generate:
  - hybrid_hypothesis
  - fusion_logic
  - innovation_points
  - expected_benefits
- pass crossover suffix to `hypothesis_agent`

Recommended selection logic:

- primary score: `Rank IC`
- secondary preference: diversity across direction and phase

### Phase 5: Parent selection and diversity scoring

Goal:

- avoid both extremes:
  - always exploit best branch
  - fully random search

Tasks:

- support:
  - `best`
  - `random`
  - `weighted`
  - `top_percent_plus_random`
- add combination score:
  - performance score
  - direction diversity
  - phase diversity
  - optional factor-expression overlap penalty

### Phase 6: Report and observability upgrade

Goal:

- make branch-level evolution visible

Tasks:

- extend markdown report with:
  - trajectory table
  - parent-child lineage
  - phase per loop
  - best branch evolution path
  - mutation/crossover summaries

This will make the system much easier to debug and compare.

## Best Fit For Our Current Codebase

Given our current architecture, the safest implementation order is:

1. add trajectory persistence
2. add controller metadata without changing agent internals
3. add mutation prompt path
4. add crossover prompt path
5. switch `main.py` from fixed loop iteration to controller-driven iteration

This is safer than starting from crossover first, because crossover without persistent trajectory state will become fragile very quickly.

## Short Recommendation

If we want maximum ROI with minimum rewrite risk, we should first build:

- `FactorTrajectory`
- `TrajectoryPool`
- `EvolutionController`

and only after that add:

- `MutationOperator`
- `CrossoverOperator`

That sequence fits our current repository much better than a big-bang rewrite.
