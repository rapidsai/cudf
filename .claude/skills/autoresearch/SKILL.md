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

**Budget**: Unlimited. Cost is not a concern — the only goal is performance improvements. Use Opus 4.6 (1M context) at maximum effort for all work including subagents. Do unlimited web searches. Don't hold back to save cost. Context auto-compacts as needed.

## Reference

Read `references/modules.md` for the CSV benchmark binaries, file lists, and results.tsv format.

## Phase 1: Setup

When starting a new experiment run:

1. **Agree on a run tag** with the user — propose one based on today's date (e.g. `apr05-csv`). Branch `autoresearch/<tag>` must not exist yet.

2. **Create branch**: `git checkout -b autoresearch/<tag>`

3. **Read in-scope files** — see `references/modules.md` for the full file list. Read every CSV source file in `cpp/src/io/csv/`. Understand the current implementation deeply before proposing any changes.

4. **Deep research phase** — do this before touching any code. Spawn **2-3 researcher agents in parallel**, each with a different focus area:
   - **Researcher 1**: CSV parsing algorithms — papers on GPU-accelerated CSV/text parsing, SIMD-style parsing, parallel field detection
   - **Researcher 2**: GPU kernel/memory optimization — coalescing patterns for text processing, shared memory for delimiter scanning, warp-level string operations
   - **Researcher 3**: Competing implementations — how cuIO, DuckDB, Apache Arrow CSV, ParaText parse CSV

   Since this is the first spawn, experiment history is empty — tell each researcher that.
   
   While researchers run in parallel, read the source code yourself to understand current algorithmic choices and trade-offs.
   
   When all researchers return, **merge and rank** all ideas into a single prioritized backlog. This is your experiment plan.

5. **Build baseline**: `build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1`

6. **Run baseline tests**:
   ```bash
   cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1
   ```

7. **Establish baseline with noise floor** — run both benchmarks **3 times** to measure variance:
   ```bash
   ./cpp/build/latest/benchmarks/CSV_READER_NVBENCH --devices 0 > baseline_reader_run1.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_READER_NVBENCH --devices 0 > baseline_reader_run2.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_READER_NVBENCH --devices 0 > baseline_reader_run3.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_WRITER_NVBENCH --devices 0 > baseline_writer_run1.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_WRITER_NVBENCH --devices 0 > baseline_writer_run2.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_WRITER_NVBENCH --devices 0 > baseline_writer_run3.log 2>&1
   ```
   Extract key metrics from all 3 runs. Record the average AND variance. Any future improvement must exceed this noise floor to count as real.

8. **Initialize results.tsv** — see `references/modules.md` for the format. Record baseline as first entry.

9. **Confirm with user**, then begin the loop.

## Phase 2: The Experiment Loop

**Run indefinitely. Never ask "should I continue?"**

### Each Iteration

1. **Objective check**: "I am optimizing the CSV parser for lower benchmark time / higher throughput as measured by `CSV_READER_NVBENCH` and `CSV_WRITER_NVBENCH`." If your change doesn't target this — pick a different idea.

2. **Hypothesize** — before writing any code, state explicitly:
   - What you're changing and why (grounded in GPU architecture or algorithm theory, or backed by a paper you found)
   - What metric you expect to improve and by roughly how much
   - What could go wrong
   
   Review results.tsv to avoid repeating failed approaches. If this is experiment 5, 10, 15, etc. — do the re-anchoring step from `.claude/rules/discipline.md` Rule 8 first.

3. **Research if needed**: Web search for papers and CUDA guides. You have unlimited search budget — use it whenever you're not confident in your hypothesis. But keep research bounded per idea — 5 minutes max, then decide: implement or skip.

4. **Implement**: Modify files in `cpp/src/` and `cpp/include/` as needed — the primary target is `cpp/src/io/csv/` but dependencies may require changes elsewhere. Don't mix unrelated ideas. Don't remove working code outside the optimization hot path.

5. **Commit**: `git add -A && git commit -m "<description>"`

6. **Build**: `build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1`
   Check: `tail -n 20 build.log` — max 3 fix attempts, then abandon.

7. **Test**: `cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1`
   Check: `tail -n 30 test.log` and `grep -c "FAILED\|PASSED" test.log`
   Tests must ALL pass — non-negotiable.

8. **Benchmark**: Run whichever benchmark is relevant to the change:
   ```bash
   ./cpp/build/latest/benchmarks/CSV_READER_NVBENCH --devices 0 > run.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_WRITER_NVBENCH --devices 0 > run.log 2>&1
   ```
   Extract: `grep -E "Elem/s|Bytes/s|GlobalMem BW|BWUtil|time" run.log`

9. **Validate results**:
   - Is the improvement larger than the baseline noise floor? If not, it's noise — discard.
   - Is the improvement >20% from a minor change? Re-run twice to confirm it's real.
   - Does the improvement hold across ALL benchmark configurations (different data types, row counts, column counts)?

10. **Record** in results.tsv (don't commit it).

11. **Decision**:
    - Improved beyond noise floor, or simpler at equal perf → **keep**
    - Within noise floor, regressed, or more complex at equal perf → **discard** with `git reset --hard HEAD~1`

12. **Clean up**: `rm -f build.log test.log run.log`

13. **Check for drift or exhaustion**:
    - **3 consecutive discards** or **3+ variations of same technique** → circuit breaker (Rule 6 & 7 in discipline.md). Time for a completely different strategy.
    - **Idea backlog running low** (fewer than 2-3 ideas left) → time to respawn researchers.
    
    When respawning researchers, **always pass the current results.tsv** so they know what's been tried. They will avoid suggesting already-failed approaches and search in genuinely new directions.

14. **Repeat**.

## CSV Parser Optimization Techniques to Consider

When researching, look for opportunities in these CSV-specific areas:
- Delimiter/newline scanning — parallel character scanning, shared memory for scan state
- Field parsing — vectorized type conversion (string→int, string→float, string→datetime)
- Memory access patterns — coalesced reads of raw CSV text, minimizing scattered writes
- Kernel fusion — combining delimiter detection + field extraction + type conversion
- Quote handling — efficient parallel handling of quoted fields with escaped characters
- Row/column decomposition — better work distribution when rows vary in length
- Data type conversion — optimizing the hot path for common types (integers, floats, strings, dates)
- Duration/datetime parsing — GPU-specific optimizations
- Warp-level primitives (shuffle, vote, match) for text processing
- Reducing warp divergence across different column types
