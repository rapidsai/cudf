# autoresearch — cuDF CSV Parser Performance Optimization

This is an experiment to have the LLM autonomously optimize the cuDF C++ CSV parser for maximum GPU performance through systematic research-driven experimentation.

## Setup

To set up a new experiment:

1. **Pick a run tag** based on today's date (e.g. `apr11-csv`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files** — read every file for full context:

   **Primary CSV source files (start here):**
   - `cpp/src/io/csv/reader_impl.cu` — CSV reader implementation
   - `cpp/src/io/csv/writer_impl.cu` — CSV writer implementation
   - `cpp/src/io/csv/csv_gpu.cu` — GPU kernel implementations for CSV parsing
   - `cpp/src/io/csv/csv_gpu.hpp` — GPU-related declarations
   - `cpp/src/io/csv/csv_common.hpp` — Common utilities and definitions
   - `cpp/src/io/csv/durations.cu` — Duration/time interval parsing on GPU
   - `cpp/src/io/csv/durations.hpp` — Duration parsing declarations
   - `cpp/src/io/csv/datetime.cuh` — DateTime parsing utilities

   You may also modify any file in `cpp/src/` and `cpp/include/` — the CSV parser has dependencies across IO utilities, common infrastructure, and type dispatching.

   **Public API headers (read for context, preserve interface contract):**
   - `cpp/include/cudf/io/csv.hpp` — Main public CSV API
   - `cpp/include/cudf/io/detail/csv.hpp` — Detail/private API

   **Benchmarks (read-only, understand what's measured):**
   - `cpp/benchmarks/io/csv/csv_reader_input.cpp` — Reader input benchmark (original)
   - `cpp/benchmarks/io/csv/csv_reader_options.cpp` — Reader options benchmark (original)
   - `cpp/benchmarks/io/csv/csv_read_realistic.cpp` — Realistic mixed-type profiles (TAXI, LOGS, ANALYTICS) — **primary**
   - `cpp/benchmarks/io/csv/csv_read_type_inference.cpp` — Type inference vs explicit dtypes — **primary**
   - `cpp/benchmarks/io/csv/csv_read_quoting.cpp` — Quoting density (0%, 25%, 100%) — **primary**
   - `cpp/benchmarks/io/csv/csv_read_scale.cpp` — Scale benchmark (256MB–4GB)
   - `cpp/benchmarks/io/csv/csv_writer.cpp` — Writer benchmark
   - `cpp/benchmarks/io/csv/csv_write_scale.cpp` — Writer scale benchmark

   **Tests (read-only, understand correctness constraints):**
   - `cpp/tests/io/csv_test.cpp` — Main CSV tests
   - `cpp/tests/streams/io/csv_test.cpp` — Stream-based CSV tests

4. **Deep research phase** — before touching any code, spawn **2-3 researcher agents in parallel**, each with a different focus:
   - **Researcher 1**: CSV parsing algorithms — papers on GPU-accelerated CSV/text parsing, SIMD-style parsing, parallel field detection
   - **Researcher 2**: GPU kernel/memory optimization — coalescing patterns for text processing, shared memory for delimiter scanning, warp-level string operations
   - **Researcher 3**: Competing implementations — how cuIO, RAPIDS, DuckDB, Apache Arrow CSV, ParaText, or other GPU databases parse CSV

   While they run, read the source code yourself. When all return, merge ideas into a ranked backlog.
5. **Build baseline**: `build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1`
6. **Run baseline tests**: `cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1`
7. **Establish noise floor** — run `eval.sh` **3 times** without code changes:
   ```bash
   ./eval.sh results/baseline_run1
   ./eval.sh results/baseline_run2
   ./eval.sh results/baseline_run3
   ```
   Compare JSON results across the 3 runs. Record the average AND variance for each benchmark. Any future improvement must exceed this noise floor to count as real.
8. **Initialize results.tsv** with the header row and baseline entry (exp 0).
9. **Initialize AGENT_LOG.md** with the run header.
10. **Begin the loop** immediately.

## What you CAN and CANNOT do

**What you CAN do:**
- Modify anything in `cpp/src/` and `cpp/include/` — the CSV parser has dependencies across the source tree (IO utilities, common data structures, type dispatching). Everything is fair game: parsing algorithms, GPU kernels, memory access patterns, thread configs, warp-level operations, shared infrastructure.
- Add new internal/detail helper functions.
- Do unlimited web searches for papers, CUDA docs, optimization guides.
- Spawn researcher agents freely for new ideas.
- Install MCP servers or Claude Code plugins on the fly if they help with analysis, profiling, or research (see "MCP / Plugin Installation" below).

**What you CANNOT do:**
- Modify `cpp/benchmarks/` — benchmarks define what is measured.
- Modify `cpp/tests/` — tests define correctness. If tests fail, the code is wrong, not the tests.
- Modify `eval.sh` — the eval script is fixed.
- Install new C++ packages or add build dependencies.
- Download or execute code from the internet — read and learn only, write all code from scratch.

## The Goal

**Optimize the CSV reader for maximum throughput on mixed-type workloads.** The primary optimization target is `CSV_READER_REALISTIC_NVBENCH` — it parses multiple data types (ints, floats, timestamps, strings) in a single `read_csv` call, which is the real-world use case. The other two primary benchmarks (`CSV_READER_TYPE_INFERENCE_NVBENCH`, `CSV_READER_QUOTING_NVBENCH`) provide coverage for type inference overhead and quoting/FSM paths.

All 3 primary benchmarks are run via `./eval.sh` after every experiment. **Every experiment must report results for ALL 3 benchmarks** — not just the one you expect to improve. A change that speeds up one benchmark but regresses another is not a win.

**Focus: CSV reader only.** The reader (`reader_impl.cu`, `csv_gpu.cu`) is the performance-critical path. Writer benchmarks are out of scope for this run.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome — that's a simplification win.

- Equal perf + simpler code = **keep**
- 1% gain + 50 lines of hacks = **probably not worth it**
- 1% gain from deleting code = **definitely keep**
- 0% gain + much simpler code = **keep**

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

## Output format

`eval.sh` saves JSON results to `results/<timestamp>/` for each benchmark. It also runs NVTX stage profiling and saves the output to `nvtx_stages.txt`.

```bash
# Primary eval (every experiment)
./eval.sh results/<tag>

# Results are in:
#   results/<tag>/CSV_READER_REALISTIC_NVBENCH.json
#   results/<tag>/CSV_READER_TYPE_INFERENCE_NVBENCH.json
#   results/<tag>/CSV_READER_QUOTING_NVBENCH.json
#   results/<tag>/nvtx_stages.txt
```

## NVTX stage profiling

`eval.sh` includes NVTX stage profiling using `nsys` on the TAXI/256MB realistic benchmark. This profiles 5 instrumented stages in the CSV reader:

- `csv::load_data_and_gather_row_offsets` — host-to-device data transfer and row offset detection
- `csv::select_data_and_row_offsets` — byte range / row range selection
- `csv::infer_column_types` — type inference pass (skipped when explicit dtypes are provided)
- `csv::determine_column_types` — final column type determination
- `csv::decode_data` — the main GPU parsing/decoding pass

Use NVTX stage timings to identify which stage dominates before forming your hypothesis. The profiling output is in `results/<tag>/nvtx_stages.txt`.

## Logging results

### results.tsv

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

Every experiment gets a sequential number. **Each experiment produces 3 rows** — one per primary benchmark. This ensures you always see the full picture, not just the benchmark you expected to improve.

The TSV has a header row and 7 columns:

```
exp	commit	metric	improvement_pct	status	benchmark	description
```

1. exp: sequential experiment number (1, 2, 3, ...)
2. commit: short git hash (7 chars)
3. metric: benchmark throughput (e.g. `1234 bytes/s` or `5.6 GiB/s`)
4. improvement_pct: vs baseline (e.g. `+5.2` or `-1.3`), `0.0` for crashes
5. status: `keep`, `discard`, `crash`, or `idea`
6. benchmark: `realistic`, `type_inference`, or `quoting`
7. description: short text of what was tried

Example:

```
exp	commit	metric	improvement_pct	status	benchmark	description
0	a1b2c3d	1234 bytes/s	0.0	keep	realistic	baseline
0	a1b2c3d	980 bytes/s	0.0	keep	type_inference	baseline
0	a1b2c3d	1100 bytes/s	0.0	keep	quoting	baseline
1	b2c3d4e	1358 bytes/s	+10.0	keep	realistic	vectorized field delimiter scanning
1	b2c3d4e	1078 bytes/s	+10.0	keep	type_inference	vectorized field delimiter scanning
1	b2c3d4e	1210 bytes/s	+10.0	keep	quoting	vectorized field delimiter scanning
2	c3d4e5f	1180 bytes/s	-4.4	discard	realistic	warp-per-row parsing
2	c3d4e5f	940 bytes/s	-4.1	discard	type_inference	warp-per-row parsing
2	c3d4e5f	1050 bytes/s	-4.5	discard	quoting	warp-per-row parsing
```

Do NOT commit results.tsv — leave it untracked.

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

This file is the long-term narrative record. results.tsv is compact metrics; AGENT_LOG.md is the reasoning. Both are untracked — do NOT commit either.

Initialize AGENT_LOG.md at the start of a run with a header:
```markdown
# Agent Log — <run tag>
# Append-only. One section per experiment.
```

## The experiment loop

LOOP FOREVER:

1. **Hypothesize** — before writing any code, state explicitly:
   - What you're changing and why (grounded in GPU architecture, algorithm theory, or a paper you found)
   - What metric you expect to improve and by roughly how much
   - What could go wrong

   Review results.tsv to avoid repeating failed approaches.

2. **Implement**: Modify files in `cpp/src/` and `cpp/include/` as needed — the primary target is `cpp/src/io/csv/` but dependencies may require changes elsewhere. One idea per experiment — don't mix unrelated changes. If an experiment with mixed ideas fails, you won't know which caused it and you've wasted a full build cycle.

3. **Commit**: `git add -A && git commit -m "<description>"`

4. **Build**: `build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1`
   Check: `tail -n 20 build.log` — if it fails, max 3 fix attempts, then abandon.

5. **Test**: `cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1`
   Check: `tail -n 30 test.log` and `grep -c "FAILED\|PASSED" test.log`
   All tests must pass — non-negotiable.

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

9. **Log to AGENT_LOG.md** — append one section for this experiment (see format in "Logging results" above). Include what worked, what didn't, what you learned, and where the research head plans to go next.

10. **Save to memory** — after each significant discovery, use `/memory` to persist insights that will be valuable in future sessions:
   - Which optimization approaches worked and why (with specific speedup numbers)
   - Which approaches failed and why (so future sessions don't repeat them)
   - Bottlenecks discovered in the CSV parser (e.g. "delimiter scanning is memory-bound, not compute-bound")
   - Architecture insights about how the parser works internally
   - Useful papers or techniques found during research
   
   Memory persists across sessions — even if context compacts or a new session starts, these notes survive. Reference memory at the start of each session and during re-anchoring to recall past discoveries.

10. **Decision**:
    - Improved beyond noise floor, or simpler at equal perf → **keep**
    - Within noise floor, regressed, or more complex at equal perf → **discard** with `git reset --hard HEAD~1`

11. **Clean up**: `rm -f build.log test.log`
    JSON results in `results/` persist across experiments — do not delete them.

12. **Discipline checks** (before next iteration):
    - **Stall detection**: 3+ experiments without improvement across ANY primary benchmark (including within-noise-floor results)? Enter **deep research phase**:
      1. STOP experimenting.
      2. Re-read source code from disk, AGENT_LOG.md, results.tsv, and NVTX stages.
      3. Spawn **2-3 deep researcher agents** with full experiment history — they must find a fundamentally new direction.
      4. Require a HIGH-confidence idea (backed by a paper or clear architectural insight) before the next experiment. Don't start another build cycle on a hunch.
    - **Force diversity**: 3+ variations of the same technique (e.g. different thread block sizes, different shared memory tile configs)? You're stuck in a local optimum. Try a completely different algorithm. The biggest gains come from algorithmic changes, not parameter sweeps.
    - **Re-anchor every 5 experiments**: Re-read results.tsv end-to-end, AGENT_LOG.md, check `/memory` for past discoveries, re-state your objective, summarize what worked and what failed, only THEN propose your next hypothesis. Long sessions cause context rot — memory is your hedge against it.
    - **Idea backlog low** (fewer than 2-3 ideas)? Respawn 2-3 researcher agents with results.tsv + AGENT_LOG.md so they know what's been tried and search in new directions.

13. **Repeat.**

**Timeout**: Build > 30 min, test > 10 min, or benchmark > 30 min → kill and treat as failure.

**Crashes**: If something dumb and easy to fix (typo, missing include), fix and re-run. If fundamentally broken, log `crash`, discard, move on.

**Output hygiene**: Always redirect build/test output to log files. Build output can be thousands of lines:
```bash
# Correct
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1
tail -n 20 build.log

# Wrong — floods context
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
```
`eval.sh` handles its own output — JSON results go to `results/` and stdout is minimal.

## Memory — Persist Discoveries Across Sessions

Claude Code has a persistent memory system (`/memory`) that survives across sessions and context compaction. Use it actively:

**When starting a session**: Check `/memory` for notes from prior sessions — which approaches were tried, what bottlenecks were found, what worked. Don't re-discover what's already known.

**During experiments**: Save significant discoveries — successful optimizations, failed approaches with reasons, architectural insights about the CSV parser, useful papers/techniques. This is especially important because context auto-compacts during long runs.

**During re-anchoring** (every 5 experiments): Read memory alongside results.tsv to get the full picture, including insights from prior sessions that aren't in the current results.tsv.

Memory complements results.tsv: the TSV tracks what was tried and the numbers; memory tracks the **why** — insights, bottleneck analysis, and strategic direction.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from the computer, and expects you to continue working **indefinitely** until you are manually stopped. You are autonomous.

If you run out of ideas, think harder:
1. Do more web searches — new papers, different search terms, adjacent GPU workloads (JSON parsing, text processing, regex matching on GPU)
2. Re-read the source code from disk looking for bottlenecks you missed
3. Try combining two previous near-miss ideas that each showed partial improvement
4. Try a fundamentally different approach (if you've been optimizing kernel launch params, try algorithmic changes; if you've been optimizing parsing, try data type conversion)
5. Spawn researcher agents to search for new techniques in parallel — always pass results.tsv so they know what's been tried

Random mutations waste build cycles. Research finds new strategies.

As an example, a user might leave you running while they sleep. Each experiment takes ~10-30 minutes (build + test + benchmark), so you can run 2-6 per hour, for a total of 15-50 overnight. The user wakes up to experimental results, all completed by you while they slept.

## Budget

Always use Opus 4.6 (1M context) at maximum effort for all work including subagents. Do unlimited web searches. Context auto-compacts as needed — don't worry about window limits.

## CSV parser optimization areas to consider

When researching, look for opportunities in these CSV-specific areas:
- **Delimiter/newline scanning**: Parallel character scanning, SIMD-style operations on GPU, shared memory for scan state
- **Field parsing**: Vectorized type conversion (string→int, string→float, string→datetime), reducing warp divergence across different column types
- **Memory access patterns**: Coalesced reads of raw CSV text, minimizing scattered writes during column extraction
- **Kernel fusion**: Combining delimiter detection + field extraction + type conversion to reduce memory round-trips
- **Quote handling**: Efficient parallel handling of quoted fields with escaped characters
- **Row/column decomposition**: Better strategies for splitting work across thread blocks when rows vary in length
- **Data type conversion**: Optimizing the hot path for common types (integers, floats, strings, dates)
- **Duration/datetime parsing**: GPU-specific optimizations in `durations.cu` and `datetime.cuh`

## MCP / Plugin Installation

You may install MCP servers or Claude Code plugins on the fly during a run if they help with analysis, profiling, visualization, or research. Examples: a JSON analysis MCP, a benchmark comparison tool, a documentation fetcher.

**If the MCP/plugin works without user intervention**: install it and use it immediately.

**If it requires user auth, API keys, or manual setup**: do NOT block on it. Instead:
1. Note the requirement in `SETUP_REQUIRED.md` (create if it doesn't exist):
   ```markdown
   ## <MCP/Plugin Name>
   - **What**: <what it does>
   - **Why**: <why it would help>
   - **Setup needed**: <what the user needs to do — auth, API key, etc.>
   - **Install command**: <the exact command to run>
   ```
2. Continue with the experiment loop — the user will set it up before the next session.

Do NOT install MCP servers that download and execute arbitrary code from the internet. Read-only data sources and analysis tools are fine.

## Quick start

To start a new experiment run, just say:

> Optimize the CSV parser for maximum throughput.

Or use the command: `/project:experiment csv`
