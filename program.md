# autoresearch — cuDF CSV Parser Performance Optimization

This is an experiment to have the LLM autonomously optimize the cuDF C++ CSV parser for maximum GPU performance through systematic research-driven experimentation.

## Run Configuration

- **Base branch**: `origin/dev_autoresearch_v2`
- **Run tag pattern**: `<monthday>-csv` (e.g. `apr12-csv`)

## Research Seeds

Starting directions for this run. These are NOT prescriptive — validate each idea against profiling data before implementing.

- (Add per-run research seeds here before starting a run)

### Known Dead Ends (do NOT retry)

- (Add known failed approaches here before starting a run)

---

## Setup

To set up a new experiment:

1. **Pick a run tag** based on today's date (e.g. `apr12-csv`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Validate and create branch**:
   ```bash
   git fetch origin
   git rev-parse --verify <base_branch> || { echo "ERROR: base branch does not exist. STOP."; exit 1; }
   git checkout -b autoresearch/<tag> <base_branch>
   ```
   where `<base_branch>` is from "Run Configuration" above. If the base branch does not exist, **STOP and tell the user** — do not fall back to a different branch.
3. **Read the in-scope files** — read every file for full context (see "File Reference" section below for the complete list). Start with the primary CSV source files in `cpp/src/io/csv/`.
4. **Read the "Research Seeds" and "Known Dead Ends" sections** at the top of this file. Use the seeds to focus initial research. Do NOT retry anything listed as a dead end.
5. **Deep research phase** — before touching any code, spawn **2-3 researcher agents in parallel**, each with a different focus:
   - **Researcher 1**: CSV parsing algorithms — papers on GPU-accelerated CSV/text parsing, SIMD-style parsing, parallel field detection
   - **Researcher 2**: GPU kernel/memory optimization — coalescing patterns for text processing, shared memory for delimiter scanning, warp-level string operations
   - **Researcher 3**: Competing implementations — how cuIO, RAPIDS, DuckDB, Apache Arrow CSV, ParaText, or other GPU databases parse CSV

   Pass the Known Dead Ends to each researcher so they avoid suggesting already-ruled-out approaches.
   While they run, read the source code yourself. When all return, merge ideas into a ranked backlog.
6. **Build baseline**: `build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1`
   If the build hangs or OOMs with `-j0` (unlimited parallelism), retry with `-j$(nproc)` or lower.
7. **Run baseline tests**: `cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1`
8. **Establish noise floor** — run `eval.sh` **3 times** without code changes:
   ```bash
   ./eval.sh results/baseline_run1
   ./eval.sh results/baseline_run2
   ./eval.sh results/baseline_run3
   ```
   Compare JSON results across the 3 runs. Record the average AND variance for each benchmark. Any future improvement must exceed this noise floor to count as real.
9. **Initialize results.tsv** with the header row and baseline entry (exp 0).
10. **Initialize AGENT_LOG.md** with the run header.
11. **Begin the loop** immediately.

---

## The Goal

**Optimize the CSV reader for maximum throughput on mixed-type workloads.** The primary optimization target is `CSV_READER_REALISTIC_NVBENCH` — it parses multiple data types (ints, floats, timestamps, strings) in a single `read_csv` call, which is the real-world use case. The other two primary benchmarks (`CSV_READER_TYPE_INFERENCE_NVBENCH`, `CSV_READER_QUOTING_NVBENCH`) provide coverage for type inference overhead and quoting/FSM paths.

All 3 primary benchmarks are run via `./eval.sh` after every experiment. **Every experiment must report results for ALL 3 benchmarks** — not just the one you expect to improve. A change that speeds up one benchmark but regresses another is not a win.

**Focus: CSV reader only.** The reader (`reader_impl.cu`, `csv_gpu.cu`) is the performance-critical path. Writer benchmarks are out of scope for this run.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome — that's a simplification win.

- Equal perf + simpler code = **keep**
- 1% gain + 50 lines of hacks = **probably not worth it**
- 1% gain from deleting code = **definitely keep**
- 0% gain + much simpler code = **keep**

---

## What you CAN and CANNOT do

**What you CAN do:**
- Modify anything in `cpp/src/` and `cpp/include/` — the CSV parser has dependencies across the source tree (IO utilities, common data structures, type dispatching). Everything is fair game: parsing algorithms, GPU kernels, memory access patterns, thread configs, warp-level operations, shared infrastructure.
- Add new internal/detail helper functions.
- Do unlimited web searches for papers, CUDA docs, optimization guides (highly recommended).
- Read code examples online to understand concepts and algorithms.
- Read API docs for CUB, Thrust, CUDA runtime (highly recommended).
- Read how other GPU databases solve similar problems.
- Spawn researcher agents freely for new ideas.
- Install MCP servers or Claude Code plugins on the fly (see "MCP / Plugin Installation" below).

**What you CANNOT do:**
- Modify `cpp/benchmarks/` — benchmarks define what is measured.
- Modify `cpp/tests/` — tests define correctness. If tests fail, the code is wrong, not the tests.
- Modify `eval.sh` — the eval script is fixed.
- Install new C++ packages or add build dependencies.
- Download or execute code from the internet — read and learn only, write all code from scratch.
- `curl | bash`, `wget`, `git clone`, `pip install`, `npm install` — or any command that downloads and runs external code.
- Copy-paste code snippets from the web and execute them as scripts.
- Add external dependencies not already in the project.

**Why the web restriction?** External code may have incompatible licenses, vulnerabilities, or dependencies that break the build. Understanding techniques and reimplementing them from scratch ensures the code fits cuDF's architecture, style, and performance constraints.

---

## Research Head Role

You are not just an experiment runner — you are the **research head**. You maintain the strategic direction of the entire optimization effort.

**Before every experiment**, act as research head:
1. Read the latest `nvtx_stages.txt` — which CSV reader stage dominates? (`load_data`, `decode_data`, `infer_column_types`, etc.)
2. Read the last few entries of `AGENT_LOG.md` — what worked, what failed, what was learned?
3. Assess: is the current optimization direction still productive, or is it time to pivot?
4. Spawn **1-2 small focused researcher agents** with a specific question (not broad surveys). Examples:
   - "The decode_data stage is 70% of runtime. Find papers on GPU-parallel type conversion for mixed int/float/string columns."
   - "Warp divergence is high in csv_gpu.cu:decode_field. Find techniques for reducing divergence when threads process different column types."
   - "The last 2 experiments targeting shared memory tiling both regressed. Find a completely different approach to memory access optimization."
5. Use their findings to form your hypothesis — only then proceed to implement.

**Key principle**: Each experiment should be informed by research, not just intuition. Small focused research before each experiment beats large unfocused research at the start. The research head always knows WHY the next experiment is worth trying.

**Researcher spawning cadence**:
- **Start of run**: 2-3 broad researchers (algorithmic survey, GPU optimization, competing implementations)
- **Each experiment**: 1-2 small focused researchers (targeted at the specific bottleneck you're attacking)
- **After stall (3+ experiments without improvement)**: 2-3 deep researchers (find a fundamentally new direction)

---

## The Experiment Loop

LOOP FOREVER:

1. **Hypothesize** — before writing any code, state explicitly:
   - What you're changing and why (grounded in GPU architecture, algorithm theory, or a paper you found)
   - What metric you expect to improve and by roughly how much
   - What could go wrong

   Review results.tsv to avoid repeating failed approaches.

2. **Implement**: Modify files in `cpp/src/` and `cpp/include/` as needed — the primary target is `cpp/src/io/csv/` but dependencies may require changes elsewhere. One idea per experiment — don't mix unrelated changes. If an experiment with mixed ideas fails, you won't know which caused it and you've wasted a full build cycle.

3. **Commit**: Stage only code files you changed, then commit. Do NOT use `git add -A` or `git add .` — these can accidentally stage untracked files like AGENT_LOG.md and results.tsv.
   ```bash
   git add cpp/src/ cpp/include/
   # If you also modified CMakeLists.txt (e.g. added a new source file):
   # git add cpp/benchmarks/CMakeLists.txt
   git commit -m "<description>"
   ```

4. **Build**: `build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1`
   Check: `tail -n 20 build.log` — if it fails, max 3 fix attempts, then abandon.
   If the build hangs or OOMs with `-j0`, retry with `-j$(nproc)` or lower.
   
   **While the build runs**, use the wait time productively:
   - Spawn 1-2 researcher agents in the background for the NEXT experiment's hypothesis
   - Read source code or profiling data to deepen understanding
   - Do NOT edit any source files while the build is in progress
   - When the build completes, proceed to Test immediately — do not wait for background research to finish. Research results feed the next experiment, not the current one.

5. **Test + Verify** (non-negotiable gate — must pass before ANY benchmarking):

   a. **Unit tests**: `cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1`
      Check: `tail -n 30 test.log` and `grep -c "FAILED\|PASSED" test.log`
      All tests must pass.

   b. **Benchmark crash check**: Run ALL CSV nvbench benchmarks in `--profile` mode (runs every config once):
      ```bash
      for f in cpp/build/latest/benchmarks/CSV_*; do
        "$f" --profile --timeout 10 > /tmp/bench_check.log 2>&1 || echo "CRASH: $(basename $f)"
        grep -qi "error\|illegal\|misaligned\|fault" /tmp/bench_check.log && echo "CUDA ERROR in $(basename $f)"
      done
      ```
      Run every binary as-is — do NOT add axis filters or sub-benchmark flags. Every benchmark must complete without CUDA errors or crashes. This catches correctness bugs that unit tests miss (e.g. issues that only manifest at scale with generated data).

   If either gate fails, the experiment is **immediately rejected**. Revert the code (see "Reverting Failed Experiments" below), log as `crash` in results.tsv and AGENT_LOG.md, and move on. Do NOT proceed to benchmarking with broken code — a CUDA error poisons the driver state for all subsequent runs.

6. **Benchmark**: Run the primary eval:
   ```bash
   ./eval.sh results/<experiment_tag>
   ```
   This runs the 3 primary benchmarks (REALISTIC, TYPE_INFERENCE, QUOTING) + NVTX stage profiling. JSON results are saved to the results directory.

   **Every 3rd experiment**, also run all CSV reader benchmarks for a holistic view:
   ```bash
   RESULTS_DIR="results/<experiment_tag>_full"
   mkdir -p "$RESULTS_DIR"
   for f in cpp/build/latest/benchmarks/CSV_READER_*; do
     name=$(basename "$f")
     "$f" --timeout 5 --json "$RESULTS_DIR/$name.json"
   done
   ```

7. **Validate results**:
   - Is the improvement larger than the noise floor? If not, it's noise — discard.
   - Is the improvement >20% from a minor change? Re-run `./eval.sh` twice more to confirm it's real.
   - Does it hold across ALL 3 primary benchmarks and their configs (different profiles, type mixes, quoting levels), or only one?
   - Check `nvtx_stages.txt` — did the expected stage speed up, or did the gain come from elsewhere?

8. **Record** in results.tsv — 3 rows for this experiment (one per primary benchmark).

9. **Log to AGENT_LOG.md** — append one section for this experiment (see "Logging" section below). Include what worked, what didn't, what you learned, and where the research head plans to go next.

10. **Save to memory** — after each significant discovery, use `/memory` to persist insights that will be valuable in future sessions:
   - Which optimization approaches worked and why (with specific speedup numbers)
   - Which approaches failed and why (so future sessions don't repeat them)
   - Bottlenecks discovered in the CSV parser (e.g. "delimiter scanning is memory-bound, not compute-bound")
   - Architecture insights about how the parser works internally
   - Useful papers or techniques found during research
   
   Memory persists across sessions — even if context compacts or a new session starts, these notes survive. Reference memory at the start of each session and during re-anchoring to recall past discoveries.

11. **Decision**:
    - Improved beyond noise floor, or simpler at equal perf → **keep**
    - Within noise floor, regressed, or more complex at equal perf → **discard** with `git revert HEAD --no-edit` (see "Reverting Failed Experiments"). NEVER use `git reset` (any mode).

12. **Clean up**: `rm -f build.log test.log`
    JSON results in `results/` persist across experiments — do not delete them.

13. **Discipline checks** (before next iteration):
    - **Stall detection**: 3+ experiments without improvement across ANY primary benchmark (including within-noise-floor results)? Enter **deep research phase**:
      1. STOP experimenting.
      2. Re-read source code from disk, AGENT_LOG.md, results.tsv, and NVTX stages.
      3. Spawn **2-3 deep researcher agents** with full experiment history — they must find a fundamentally new direction.
      4. Require a HIGH-confidence idea (backed by a paper or clear architectural insight) before the next experiment. Don't start another build cycle on a hunch.
    - **Force diversity**: 3+ variations of the same technique (e.g. different thread block sizes, different shared memory tile configs)? You're stuck in a local optimum. Try a completely different algorithm. The biggest gains come from algorithmic changes, not parameter sweeps.
    - **Re-anchor every 5 experiments**: Re-read results.tsv end-to-end, AGENT_LOG.md, check `/memory` for past discoveries, re-state your objective, summarize what worked and what failed, only THEN propose your next hypothesis. Long sessions cause context rot — memory is your hedge against it.
    - **Idea backlog low** (fewer than 2-3 ideas)? Respawn 2-3 researcher agents with results.tsv + AGENT_LOG.md so they know what's been tried and search in new directions.

14. **Repeat.**

---

## Output Format and Logging

### eval.sh output

`eval.sh` saves JSON results to `results/<timestamp>/` for each benchmark. It also runs NVTX stage profiling.

```bash
# Primary eval (every experiment)
./eval.sh results/<tag>

# Results are in:
#   results/<tag>/CSV_READER_REALISTIC_NVBENCH.json
#   results/<tag>/CSV_READER_TYPE_INFERENCE_NVBENCH.json
#   results/<tag>/CSV_READER_QUOTING_NVBENCH.json
#   results/<tag>/nvtx_stages.txt
```

### NVTX stage profiling

`eval.sh` includes NVTX stage profiling using `nsys` on the TAXI/256MB realistic benchmark. This profiles 5 instrumented stages in the CSV reader:

- `csv::load_data_and_gather_row_offsets` — host-to-device data transfer and row offset detection
- `csv::select_data_and_row_offsets` — byte range / row range selection
- `csv::infer_column_types` — type inference pass (skipped when explicit dtypes are provided)
- `csv::determine_column_types` — final column type determination
- `csv::decode_data` — the main GPU parsing/decoding pass

Use NVTX stage timings to identify which stage dominates before forming your hypothesis.

### results.tsv

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

Every experiment gets a sequential number. **Each experiment produces 3 rows** — one per primary benchmark.

The TSV has a header row and 7 columns:

```
exp	commit	metric	improvement_pct	status	benchmark	description
```

1. exp: sequential experiment number (0 = baseline, 1, 2, 3, ...)
2. commit: short git hash (7 chars)
3. metric: benchmark throughput (e.g. `5.2 GiB/s`). For `realistic`, include per-type breakdown if useful: `5.2 GiB/s (INT:7.1 FLT:6.9 TS:6.0)`
4. improvement_pct: vs baseline (e.g. `+5.2` or `-1.3`), `0.0` for crashes
5. status: `keep`, `discard`, `crash`, or `idea`
6. benchmark: `realistic`, `type_inference`, or `quoting`
7. description: short text of what was tried

results.tsv is **untracked and append-only**. Never edit or delete existing rows. Because it is untracked, you must ensure code reverts never destroy it — see "Reverting Failed Experiments" below.

### AGENT_LOG.md

Append-only narrative log. After every experiment, append one section:

```markdown
## Experiment N: <short title>

**Hypothesis**: <what you changed and why>
**Result**: <keep/discard/crash> — <one-line summary of numbers>

### What worked
- <bullet points>

### What didn't
- <bullet points>

### What I learned
- <bullet points — insights about the parser, GPU behavior, or algorithm>

### Next direction
- <what the research head plans to try next and why>
```

This file is the long-term narrative record. results.tsv is compact metrics; AGENT_LOG.md is the reasoning.

Initialize AGENT_LOG.md at the start of a run with:
```markdown
# Agent Log — <run tag>
# Append-only. One section per experiment.
```

**Both files are untracked and append-only.** Never edit or delete existing content — only append new entries. Because they are untracked, they are vulnerable to destructive git operations. This is why `git reset` is strictly forbidden.

---

## Benchmark Reference

### Primary benchmarks (run via `eval.sh` every experiment)

| Benchmark | Binary Name | What It Measures |
|---|---|---|
| Realistic mixed-type profiles | CSV_READER_REALISTIC_NVBENCH | TAXI (14 cols), LOGS (6 cols), ANALYTICS (8 cols) at 256/512/1024 MB |
| Type inference vs explicit | CSV_READER_TYPE_INFERENCE_NVBENCH | With/without inference across ALL_INTEGRAL, ALL_FLOAT, ALL_STRING, MIXED |
| Quoting density | CSV_READER_QUOTING_NVBENCH | 0%, 25%, 100% quoted columns at 64/256 MB |

### Holistic benchmarks (run every 3 experiments)

| Benchmark | Binary Name | What It Measures |
|---|---|---|
| Original reader (input sizes) | CSV_READER_NVBENCH | Various input sizes and formats |
| Scale (large data) | CSV_READER_SCALE_NVBENCH | Mixed-type from 256 MB to 4 GB |

---

## File Reference

### Primary target (CSV parser source — start here)
- `cpp/src/io/csv/reader_impl.cu` — CSV reader implementation
- `cpp/src/io/csv/writer_impl.cu` — CSV writer implementation
- `cpp/src/io/csv/csv_gpu.cu` — GPU kernel implementations for CSV parsing
- `cpp/src/io/csv/csv_gpu.hpp` — GPU-related declarations and templates
- `cpp/src/io/csv/csv_common.hpp` — Common utilities and definitions
- `cpp/src/io/csv/durations.cu` — Duration/time interval parsing on GPU
- `cpp/src/io/csv/durations.hpp` — Duration parsing declarations
- `cpp/src/io/csv/datetime.cuh` — DateTime parsing utilities (CUDA header)

### Editable zone
- `cpp/src/` — all source files (CSV has dependencies across IO utilities, common infrastructure, type dispatching)
- `cpp/include/` — all headers (may need to modify internal/detail headers for optimization)

### Public API headers (preserve interface contract)
- `cpp/include/cudf/io/csv.hpp` — Main public CSV API (readers/writers)
- `cpp/include/cudf/io/detail/csv.hpp` — Implementation details, private API

### Benchmarks (read-only — understand what's measured)
- `cpp/benchmarks/io/csv/csv_reader_input.cpp` — Reader benchmark with different input sizes/formats
- `cpp/benchmarks/io/csv/csv_reader_options.cpp` — Reader benchmark with various options
- `cpp/benchmarks/io/csv/csv_read_realistic.cpp` — Realistic mixed-type profiles (TAXI, LOGS, ANALYTICS) **primary**
- `cpp/benchmarks/io/csv/csv_read_type_inference.cpp` — Type inference vs explicit dtypes **primary**
- `cpp/benchmarks/io/csv/csv_read_quoting.cpp` — Quoting density (0%, 25%, 100%) **primary**
- `cpp/benchmarks/io/csv/csv_read_scale.cpp` — Scale benchmark (256MB–4GB)
- `cpp/benchmarks/io/csv/csv_writer.cpp` — Writer benchmark
- `cpp/benchmarks/io/csv/csv_write_scale.cpp` — Writer scale benchmark

### Tests (read-only — understand correctness constraints)
- `cpp/tests/io/csv_test.cpp` — Main CSV reader/writer tests
- `cpp/tests/streams/io/csv_test.cpp` — Stream-based CSV tests

---

## Experiment Safety Rules

### Read-Only Zones (Non-Negotiable)

These define ground truth. Modifying them invalidates all experiment results.

- **`cpp/benchmarks/**`** — Benchmarks define what is measured. Never modify.
- **`cpp/tests/**`** — Tests define correctness. If tests fail, the code is wrong, not the tests.
- **`eval.sh`** — The eval script is fixed. Never modify.

### API Contract

Public API function signatures should be preserved — do not change existing public function signatures or remove public functions/types. Adding new internal/detail helpers and new overloads that don't break existing callers is fine.

### Build System

- Do NOT modify `CMakeLists.txt` unless strictly necessary for new source files you've added.
- Do NOT install new packages or add dependencies beyond what's in `pyproject.toml` / `CMakeLists.txt`.

### Output Hygiene

Always redirect command output to log files. Build output can be thousands of lines and will flood context:

```bash
# Correct
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1
tail -n 20 build.log

# Wrong — floods context
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
```

Clean up logs after each experiment: `rm -f build.log test.log run.log`

### Reverting Failed Experiments

**NEVER use `git reset` (any mode: `--hard`, `--soft`, `--mixed`) to undo an experiment.** This rewrites commit history and can destroy untracked files like AGENT_LOG.md and results.tsv.

**NEVER use `git rebase`, `git checkout .`, or any other history-rewriting or bulk-restore command.**

To discard a failed experiment:
```bash
git revert HEAD --no-edit
```
This creates a new commit that is the exact inverse of the experiment. It handles modified files, new files, and deleted files automatically. Since experiment commits only contain code files (`cpp/src/`, `cpp/include/`), the revert only touches code — untracked files (AGENT_LOG.md, results.tsv) are untouched.

This produces a clean forward-only history: experiment commit → revert commit. No commits are ever lost.

**Append-only files (untracked but must be preserved):**
- `AGENT_LOG.md` — never edit or delete existing entries, only append
- `results.tsv` — never edit or delete existing rows, only append
- `results/` directory — never delete previous benchmark results

These files survive code reverts because the revert procedure only touches `cpp/src/` and `cpp/include/` files. The reason `git reset` is forbidden is precisely because it CAN destroy these untracked files.

### Timeouts

- Build > 30 minutes → something is wrong, kill and investigate
- Test binary > 10 minutes → kill (`pkill -f ctest`), treat as failure
- Benchmark > 30 minutes → kill

---

## Discipline Rules

### Rule 1 — Stay on objective
Every experiment must directly target the optimization goal for the current module. If you're thinking "this is interesting, let me explore..." — STOP. Log the idea in results.tsv as an `idea` status row and move on.

### Rule 2 — One idea per experiment, clear numbering
Related changes (e.g. a new algorithm + the memory layout it requires) belong together. Unrelated ideas must be separate experiments. Each build-test-benchmark cycle costs 10-30 minutes — if an experiment with mixed ideas fails, you won't know which idea caused it and you've wasted that entire cycle.

**Naming**: Experiments are numbered sequentially: Exp1, Exp2, Exp3, etc. This is the ONLY numbering system. Researcher agents produce ranked idea lists (e.g. "Idea A", "Idea B" or "Tier 1, Tier 2") — these are NOT experiment numbers. An idea becomes an experiment only when you commit code for it and assign the next Exp number. Fix-up commits within an experiment keep the same number (e.g. "Exp6: fix compile error" is still Exp6, not Exp7).

### Rule 3 — Establish baseline noise floor
On your first run, run the baseline benchmark **3 times** without code changes. Record the variance in key metrics (Elem/s, time). Any future improvement smaller than this variance is benchmark noise, not a real improvement. Do not keep changes within the noise floor.

### Rule 4 — Verify surprising results
If a result looks too good (>20% gain from a minor change), re-run the benchmark twice more. NVBench measurements have natural variance from GPU thermal throttling, memory allocation timing, and other factors. Only trust reproducible numbers across multiple runs.

### Rule 5 — Clean state between experiments
After every experiment (keep or discard), delete temporary files: `rm -f build.log test.log run.log`. Stale logs from prior experiments accumulate in context and contribute to drift over many iterations. To discard a failed experiment's code, follow the revert procedure above — never use `git reset` (any mode).

### Rule 6 — Stall detection and deep research phase
If 3+ experiments show no improvement across ANY primary benchmark (including within-noise-floor results, not just `discard`/`crash`), you are stalled. STOP experimenting and enter **deep research phase**:
1. Re-read the source code from disk (not from memory — your cached understanding may be wrong after many edits)
2. Re-read results.tsv AND AGENT_LOG.md end-to-end
3. Read NVTX stage profiling — reassess which stage is the actual bottleneck
4. Spawn **2-3 deep researcher agents** with the full experiment history. They must find a fundamentally new direction.
5. Require a **high-confidence idea** (backed by a paper, clear architectural insight, or profiling data) before starting the next experiment. Don't spend another build cycle on a hunch.

### Rule 7 — Force diversity after local optima
If you've made 3+ variations of the same technique (e.g. different shared memory tile sizes, different thread block dimensions, different radix widths), you are stuck in a local optimum. STOP tuning parameters and try a completely different algorithmic approach. The biggest GPU performance gains come from algorithmic changes and memory access pattern redesigns, not parameter sweeps.

### Rule 8 — Research head: assess before every hypothesis
Before forming each hypothesis, act as research head:
1. Read the latest `nvtx_stages.txt` — which stage dominates?
2. Read the last entries of `AGENT_LOG.md` — what patterns are emerging?
3. Spawn 1-2 small focused researcher agents targeting the specific bottleneck.
4. Only then form a research-backed hypothesis.
Each experiment should be informed by targeted research, not just intuition from the previous result.

### Rule 9 — Re-anchor every 5 experiments (prevents context rot)
Every 5th experiment, do ALL of the following — in long sessions, instructions loaded early fade from active attention as context grows:
1. Re-read this file (`program.md`) from disk
2. Re-read results.tsv and AGENT_LOG.md end-to-end
3. Check `/memory` for discoveries from prior sessions and earlier in this session
4. Re-state your objective in one sentence
5. Summarize which approaches worked, which failed, and why
6. Save any new insights to memory that haven't been persisted yet
7. Only THEN propose your next hypothesis

### Rule 10 — No scope creep
Do not "also optimize" a neighboring module. Do not refactor code outside the critical path. Do not add logging, profiling instrumentation, or utility functions unless directly required for the current experiment.

### Rule 11 — Write your hypothesis before writing code
Before implementing, state your hypothesis explicitly:
- What you're changing and why (grounded in GPU architecture or algorithm theory)
- What metric you expect to improve and by roughly how much
- What could go wrong

Each build cycle is expensive. This forces you to think before spending 10-30 minutes on a doomed idea.

### Rule 12 — Don't strip working code outside the hot path
If existing code handles edge cases, error paths, or special types that the benchmark doesn't exercise, leave it alone. Removing working functionality to simplify the benchmark path is metric gaming, not optimization. The benchmarks test common cases; the tests verify correctness for all cases.

### Rule 13 — Use memory to persist discoveries
After each significant finding (successful optimization, failed approach with clear reason, bottleneck identified, useful paper found), save it to `/memory`. Memory survives across sessions and context compaction. At the start of each session, check memory for prior discoveries — don't re-try approaches that already failed or re-discover known bottlenecks.

### Quality over quantity
Every rejected experiment wastes a full build-test-benchmark cycle (10-30 min). A well-reasoned hypothesis with high confidence is worth more than five speculative shots.

### When you run out of ideas
This is NOT the time to make random changes to GPU kernel parameters. Instead:
1. Do more web searches — new papers, different search terms, adjacent GPU workloads (JSON parsing, text processing, regex matching on GPU)
2. Re-read the source code from disk looking for bottlenecks you missed
3. Try combining two previous near-miss ideas that each showed partial improvement
4. Try a fundamentally different approach (if you've been optimizing kernel launch params, try algorithmic changes; if you've been optimizing parsing, try data type conversion)
5. Spawn researcher agents to search for new techniques in parallel — always pass results.tsv so they know what's been tried

Random mutations waste build cycles. Research finds new strategies.

---

## MCP / Plugin Installation

You may install MCP servers or CLI plugins on the fly during a run if they help with analysis, profiling, visualization, or research. Examples: a JSON analysis MCP, a benchmark comparison tool, a documentation fetcher.

### Allowed
- Install well-known, trusted MCP servers (e.g., documentation lookup, profiling tools, analysis helpers)
- Install CLI tools available via system package managers if they aid benchmarking or profiling
- Configure MCP servers in `.claude/settings.json` or equivalent

### When User Intervention Is Needed

If an MCP server or plugin requires **user authentication or manual setup** (OAuth tokens, API keys, SSH keys, browser-based auth, etc.):
1. Note the requirement in `SETUP_REQUIRED.md` (create if it doesn't exist):
   ```markdown
   ## <MCP/Plugin Name>
   - **What**: <what it does>
   - **Why**: <why it would help>
   - **Setup needed**: <what the user needs to do — auth, API key, etc.>
   - **Install command**: <the exact command to run>
   ```
2. Continue with the experiment loop — the user will set it up before the next session.

### Forbidden
- Installing random or unverified MCP servers from unknown sources
- Any MCP/plugin that downloads and executes external optimization code
- Anything that violates the web research read-only rule above

---

## Memory — Persist Discoveries Across Sessions

Claude Code has a persistent memory system (`/memory`) that survives across sessions and context compaction. Use it actively:

**When starting a session**: Check `/memory` for notes from prior sessions — which approaches were tried, what bottlenecks were found, what worked. Don't re-discover what's already known. However, evaluate whether prior memories still apply — if the base branch has changed since the last session, prior profiling data and bottleneck analysis may be stale. Re-profile before trusting old numbers.

**During experiments**: Save significant discoveries — successful optimizations, failed approaches with reasons, architectural insights about the CSV parser, useful papers/techniques. This is especially important because context auto-compacts during long runs.

**During re-anchoring** (every 5 experiments): Read memory alongside results.tsv to get the full picture, including insights from prior sessions that aren't in the current results.tsv.

Memory complements results.tsv: the TSV tracks what was tried and the numbers; memory tracks the **why** — insights, bottleneck analysis, and strategic direction.

---

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from the computer, and expects you to continue working **indefinitely** until you are manually stopped. You are autonomous.

As an example, a user might leave you running while they sleep. Each experiment takes ~10-30 minutes (build + test + benchmark), so you can run 2-6 per hour, for a total of 15-50 overnight. The user wakes up to experimental results, all completed by you while they slept.

## Budget

Always use Opus 4.6 (1M context) at maximum effort for all work including subagents. Do unlimited web searches. Context auto-compacts as needed — don't worry about window limits.

---

## WARNING: Micro-benchmark tunnel vision

**The single-type benchmarks can mislead optimization direction.** Each single-type benchmark measures one data type in isolation with clean, pre-typed, large data — essentially the best case. This can cause the agent to:

1. **Write specialized per-type kernels** instead of optimizing the common mixed-type path
2. **Tunnel-vision on one technique** (e.g. "fused delimiter scan + type conversion" for every type)
3. **Miss cross-type interactions** — warp divergence when threads in the same warp process different column types, register pressure from multiple type-specific code paths, etc.

**The REALISTIC benchmark (TAXI/LOGS/ANALYTICS) is the ground truth** because real CSV files have mixed types. An optimization that looks great on single-type benchmarks may not help (or may hurt) the mixed-type case due to instruction cache pressure, register spilling, or warp divergence.

**If you find yourself applying the same technique to each type one by one, STOP.** You are optimizing micro-benchmarks, not real-world performance. Step back and think about what matters for the mixed-type workload.

## Research: real-world CSV characteristics

During research phases, understand how **real-world CSVs differ from synthetic benchmarks.** The NVBench suite uses clean, pre-typed, uniform data. Real CSV files have:
- Mixed types in the same file (the common case)
- Type inference overhead (not pre-typed)
- Quoted fields with special characters
- Variable-length string columns
- NULL/NA values scattered throughout
- Headers, comments, BOM markers

Keep this gap in mind when evaluating optimization ideas — a technique that wins on clean synthetic data may not help on messy real data. Actively search for GPU-accelerated CSV/text parsing papers and implementations, and compare cuDF's approach to what others do — especially how they handle mixed types, type inference, and quoted fields on GPU.

---

## CSV Parser Optimization Areas to Consider

Prioritize by impact tier — architecture-level changes have the highest payoff.

**Architecture-level (highest impact, explore FIRST):**
- **Mixed-type warp divergence**: When threads in a warp process columns of different types, divergence kills throughput. Can we reorganize work to reduce this?
- **Multi-pass vs single-pass architecture**: Is the current multi-kernel approach (row detection → field detection → type conversion) optimal, or can phases be overlapped/fused at a higher level?
- **Memory bandwidth utilization**: The raw CSV data is read multiple times across kernels. Can we reduce total memory traffic?
- **Host-side overhead**: Memory allocation, column construction, metadata processing — what fraction of wall time is non-GPU?
- **Multi-stream pipelining**: For large files, overlap H2D transfer with compute
- **Efficient Scan algorithm usage**: whenever you are looking into scan based algorithms, use efficient scan algorithms like CUB decoupled lookback algorithm (Example available in cccl repository), and similarly other algorithms also efficient one available via research. If any part of the algorithm seems useful, pick them up too and adapt for existing code and algorithm.

**Kernel-level (moderate impact):**
- **Delimiter/newline scanning**: Parallel character scanning, SIMD-style operations on GPU, shared memory for scan state
- **Field parsing**: Vectorized type conversion (string→int, string→float, string→datetime)
- **Memory access patterns**: Coalesced reads of raw CSV text, minimizing scattered writes during column extraction
- **Kernel fusion**: Combining delimiter detection + field extraction + type conversion to reduce memory round-trips
- **Quote handling**: Efficient parallel handling of quoted fields with escaped characters
- **Row/column decomposition**: Better strategies for splitting work across thread blocks when rows vary in length

**Type-specific (lower impact, only after architecture is optimized):**
- **Data type conversion**: Optimizing the hot path for common types (integers, floats, strings, dates)
- **Duration/datetime parsing**: GPU-specific optimizations in `durations.cu` and `datetime.cuh`

---

## Quick start

To start a new experiment run, just say:

> Optimize the CSV parser for maximum throughput.

Or use the command: `/project:experiment csv`
