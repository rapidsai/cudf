---
name: autoresearch
description: >
  Autonomous cuDF C++ CSV parser performance optimization through research-driven experimentation.
  Use this skill whenever the user mentions optimizing the CSV parser, running CSV performance
  experiments, benchmarking CSV reading/writing, improving CSV GPU kernel performance, or starting
  an autoresearch session. Also trigger when the user says things like "optimize", "benchmark",
  "experiment", "speed up", "make faster", or references program.md.
---

# Autoresearch: Autonomous cuDF C++ CSV Parser Optimization

You are an autonomous research agent that optimizes the cuDF C++ CSV parser for maximum GPU performance through systematic experimentation. You run indefinitely until manually stopped.

**`program.md` is the source of truth.** Read it first. This skill reinforces the workflow; constraints and discipline rules are enforced by `.claude/rules/`.

**Goal** is performance improvements. Use Opus 4.6 (1M context) at maximum effort for all work including subagents. Do unlimited web searches. Don't hold back to save cost. Context auto-compacts as needed.

## Reference

Read `references/modules.md` for the CSV benchmark binaries, file lists, and results.tsv format.

## Phase 1: Setup

When starting a new experiment run:

1. **Pick a run tag** based on today's date (e.g. `apr11-csv`). Branch `autoresearch/<tag>` must not exist yet.

2. **Follow the Setup steps in `program.md`** (steps 2–11). That is the source of truth for branch creation, validation, reading seeds/dead-ends, research, baseline, and initialization. Do not skip any step.

5. **Build baseline**: `build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1`

6. **Run baseline tests**:
   ```bash
   cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1
   ```

7. **Establish baseline with noise floor** — run `eval.sh` **3 times** to measure variance:
   ```bash
   ./eval.sh results/baseline_run1
   ./eval.sh results/baseline_run2
   ./eval.sh results/baseline_run3
   ```
   Compare JSON results across the 3 runs. Record the average AND variance for each benchmark. Any future improvement must exceed this noise floor to count as real. Also review `nvtx_stages.txt` to understand which CSV reader stages dominate baseline time.

8. **Initialize results.tsv** — see `references/modules.md` for the format. Record baseline as exp 0 (3 rows, one per benchmark).

9. **Initialize AGENT_LOG.md** — create with run header. This is the append-only narrative log.

10. **Begin the loop** immediately.

## Phase 2: The Experiment Loop

**Run indefinitely. Never ask "should I continue?"**

**You are the research head.** You maintain the strategic direction across all experiments. Before each experiment:
1. Read the latest `nvtx_stages.txt` — which CSV reader stage dominates?
2. Read the last few entries of `AGENT_LOG.md` — what worked, what failed, what was learned?
3. Spawn **1-2 small focused researcher agents** with a specific question (not broad surveys) targeting the bottleneck you're attacking.
4. Use their findings to form your hypothesis — only then implement.

Each experiment should be research-informed, not just intuition. Small focused research per experiment beats large unfocused research at the start.

### Each Iteration

1. **Objective check**: "I am optimizing the CSV reader for maximum throughput on mixed-type workloads. The primary target is `CSV_READER_REALISTIC_NVBENCH` (TAXI/LOGS/ANALYTICS). All 3 primary benchmarks must be measured for every experiment via `./eval.sh`." If your change doesn't target this — pick a different idea.

2. **Hypothesize** — before writing any code, state explicitly:
   - What you're changing and why (grounded in GPU architecture or algorithm theory, or backed by a paper you found)
   - What metric you expect to improve and by roughly how much
   - What could go wrong
   
   Review results.tsv to avoid repeating failed approaches. If this is experiment 5, 10, 15, etc. — do the re-anchoring step from `.claude/rules/discipline.md` Rule 8 first.

3. **Research if needed**: Web search for papers and CUDA guides. You have unlimited search budget — use it whenever you're not confident in your hypothesis. But keep research bounded per idea — 5 minutes max, then decide: implement or skip.

4. **Implement**: Modify files in `cpp/src/` and `cpp/include/` as needed — the primary target is `cpp/src/io/csv/` but dependencies may require changes elsewhere. Don't mix unrelated ideas. Don't remove working code outside the optimization hot path.

5. **Commit**: `git add cpp/src/ cpp/include/ && git commit -m "<description>"`
   Only stage code files. Do NOT use `git add -A`.

6. **Build**: `build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1`
   Check: `tail -n 20 build.log` — max 3 fix attempts, then abandon.

7. **Test + Verify** (non-negotiable gate):

   a. **Unit tests**: `cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1`
      All tests must pass.

   b. **Benchmark crash check**: Run ALL CSV nvbench benchmarks in `--profile` mode (every config once):
      ```bash
      for f in cpp/build/latest/benchmarks/CSV_*; do
        "$f" --profile --timeout 10 > /tmp/bench_check.log 2>&1 || echo "CRASH: $(basename $f)"
        grep -qi "error\|illegal\|misaligned\|fault" /tmp/bench_check.log && echo "CUDA ERROR in $(basename $f)"
      done
      ```
      Run every binary as-is — do NOT add axis filters or flags. If any crash, the experiment is immediately rejected — revert code, log as `crash`, move on.

8. **Benchmark**: Run the primary eval:
   ```bash
   ./eval.sh results/<experiment_tag>
   ```
   This runs the 3 primary benchmarks (REALISTIC, TYPE_INFERENCE, QUOTING) + NVTX stage profiling. JSON results are saved to the results directory.

   **Every 3rd experiment**, also run all CSV reader benchmarks for a holistic view:
   ```bash
   RESULTS_DIR="results/<experiment_tag>_full" && mkdir -p "$RESULTS_DIR"
   for f in cpp/build/latest/benchmarks/CSV_READER_*; do
     name=$(basename "$f")
     "$f" --timeout 5 --json "$RESULTS_DIR/$name.json"
   done
   ```

9. **Validate results**:
   - Is the improvement larger than the baseline noise floor? If not, it's noise — discard.
   - Is the improvement >20% from a minor change? Re-run twice to confirm it's real.
   - Does the improvement hold across ALL benchmark configurations (different data types, row counts, column counts)?

10. **Record** in results.tsv — 3 rows for this experiment (one per primary benchmark). Do not commit it (untracked), but never delete or edit existing rows.

11. **Log to AGENT_LOG.md** — append one section for this experiment: hypothesis, result, what worked, what didn't, what you learned, and the research head's planned next direction.

12. **Decision**:
    - Improved beyond noise floor, or simpler at equal perf → **keep**
    - Within noise floor, regressed, or more complex at equal perf → **discard** with `git revert HEAD --no-edit` (see experiment-safety.md). NEVER use `git reset` (any mode).

13. **Clean up**: `rm -f build.log test.log`
    JSON results in `results/` persist across experiments — do not delete them.

14. **Check for drift or exhaustion**:
    - **Stall detection**: 3+ experiments without improvement across ANY primary benchmark? Enter **deep research phase** — STOP experimenting, spawn 2-3 deep researchers with full history (results.tsv + AGENT_LOG.md), require a high-confidence idea before next experiment.
    - **Force diversity**: 3+ variations of same technique → circuit breaker (Rule 6 & 7 in discipline.md). Try a completely different algorithm.
    - **Idea backlog running low** (fewer than 2-3 ideas left) → time to respawn researchers.
    
    When respawning researchers, **always pass results.tsv + AGENT_LOG.md** so they know what's been tried and what was learned. They will avoid suggesting already-failed approaches and search in genuinely new directions.

15. **Repeat**.

## CSV Parser Optimization Techniques to Consider

Prioritize by impact tier — architecture-level changes have the highest payoff.

**Architecture-level (highest impact, explore FIRST):**
- Mixed-type warp divergence — reorganize work to reduce divergence when threads process different column types
- Multi-pass vs single-pass — can phases (row detect → field detect → type convert) be overlapped/fused?
- Memory bandwidth — reduce total memory traffic (raw CSV data is read multiple times across kernels)
- Host-side overhead — memory allocation, column construction, metadata processing
- Multi-stream pipelining — overlap H2D transfer with compute for large files

**Kernel-level (moderate impact):**
- Delimiter/newline scanning — parallel character scanning, shared memory for scan state
- Field parsing — vectorized type conversion (string→int, string→float, string→datetime)
- Memory access patterns — coalesced reads of raw CSV text, minimizing scattered writes
- Kernel fusion — combining delimiter detection + field extraction + type conversion
- Quote handling — efficient parallel handling of quoted fields with escaped characters

**Type-specific (lower impact, only after architecture is optimized):**
- Data type conversion — optimizing the hot path for common types
- Duration/datetime parsing — GPU-specific optimizations

**WARNING**: If you find yourself applying the same technique to each type one by one, STOP. You are optimizing micro-benchmarks, not the mixed-type workload. See "Micro-benchmark tunnel vision" warning in program.md.
