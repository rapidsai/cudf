# autoresearch — cuDF CSV Parser Performance Optimization

This is an experiment to have the LLM autonomously optimize the cuDF C++ CSV parser for maximum GPU performance through systematic research-driven experimentation.

## Setup

To set up a new experiment (decide everything autonomously — never ask the user):

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr05-csv`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
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
   - `cpp/benchmarks/io/csv/csv_reader_input.cpp` — Reader input benchmark
   - `cpp/benchmarks/io/csv/csv_reader_options.cpp` — Reader options benchmark
   - `cpp/benchmarks/io/csv/csv_writer.cpp` — Writer benchmark

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
7. **Establish noise floor** — run both benchmarks **3 times** without code changes:
   ```bash
   ./cpp/build/latest/benchmarks/CSV_READER_NVBENCH --devices 0 > baseline_reader_run1.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_READER_NVBENCH --devices 0 > baseline_reader_run2.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_READER_NVBENCH --devices 0 > baseline_reader_run3.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_WRITER_NVBENCH --devices 0 > baseline_writer_run1.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_WRITER_NVBENCH --devices 0 > baseline_writer_run2.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_WRITER_NVBENCH --devices 0 > baseline_writer_run3.log 2>&1
   ```
   Extract key metrics from all 3 runs of each. Record the average AND variance. Any future improvement must exceed this noise floor to count as real.
8. **Initialize results.tsv** with the header row and baseline entry.
9. **Begin the loop**: Setup is done — immediately start the experiment loop. Do not pause or ask for confirmation.

## What you CAN and CANNOT do

**What you CAN do:**
- Modify anything in `cpp/src/` and `cpp/include/` — the CSV parser has dependencies across the source tree (IO utilities, common data structures, type dispatching). Everything is fair game: parsing algorithms, GPU kernels, memory access patterns, thread configs, warp-level operations, shared infrastructure.
- Add new internal/detail helper functions.
- Do unlimited web searches for papers, CUDA docs, optimization guides.
- Spawn researcher agents freely for new ideas.

**What you CANNOT do:**
- Modify `cpp/benchmarks/` — benchmarks define what is measured.
- Modify `cpp/tests/` — tests define correctness. If tests fail, the code is wrong, not the tests.
- Install new packages or add dependencies.
- Download or execute code from the internet — read and learn only, write all code from scratch.

## The Goal

**Get the highest throughput / lowest time for the CSV reader and writer as measured by `CSV_READER_NVBENCH` and `CSV_WRITER_NVBENCH`.** Everything is fair game: change parsing algorithms, GPU kernel implementations, memory access patterns, thread configurations, data type conversion. The only constraint is that all tests pass.

**Primary focus: CSV reader.** The reader (`reader_impl.cu`, `csv_gpu.cu`) is the performance-critical path — CSV writing is simpler and less commonly benchmarked. Prioritize reader optimizations, but don't ignore writer wins if they're easy.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome — that's a simplification win.

- Equal perf + simpler code = **keep**
- 1% gain + 50 lines of hacks = **probably not worth it**
- 1% gain from deleting code = **definitely keep**
- 0% gain + much simpler code = **keep**

## Output format

Benchmarks print NVBench results. Extract key metrics from the log:

```bash
grep -E "Elem/s|Bytes/s|GlobalMem BW|BWUtil|time" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	metric	improvement_pct	status	benchmark	description
```

1. commit: short git hash (7 chars)
2. metric: primary benchmark number (e.g. `1234 Elem/s` or `5.6 GiB/s`)
3. improvement_pct: vs baseline (e.g. `+5.2` or `-1.3`), `0.0` for crashes
4. status: `keep`, `discard`, `crash`, or `idea`
5. benchmark: which benchmark (`csv_reader` or `csv_writer`)
6. description: short text of what was tried

Example:

```
commit	metric	improvement_pct	status	benchmark	description
a1b2c3d	1234 Elem/s	0.0	keep	csv_reader	baseline
b2c3d4e	1358 Elem/s	+10.0	keep	csv_reader	vectorized field delimiter scanning with shared memory
c3d4e5f	1180 Elem/s	-4.4	discard	csv_reader	warp-per-row parsing (too much divergence)
d4e5f6g	0	0.0	crash	csv_reader	custom SIMD-style parser (compile error)
```

Do NOT commit results.tsv — leave it untracked.

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

6. **Benchmark**: Run whichever benchmark is relevant to the change:
   ```bash
   ./cpp/build/latest/benchmarks/CSV_READER_NVBENCH --devices 0 > run.log 2>&1
   ./cpp/build/latest/benchmarks/CSV_WRITER_NVBENCH --devices 0 > run.log 2>&1
   ```
   Extract: `grep -E "Elem/s|Bytes/s|GlobalMem BW|BWUtil|time" run.log`

7. **Validate results**:
   - Is the improvement larger than the noise floor? If not, it's noise — discard.
   - Is the improvement >20% from a minor change? Re-run twice to confirm it's real.
   - Does it hold across ALL benchmark configs (different data types, row counts, column counts), or only one?

8. **Record** in results.tsv.

9. **Save to memory** — after each significant discovery, use `/memory` to persist insights that will be valuable in future sessions:
   - Which optimization approaches worked and why (with specific speedup numbers)
   - Which approaches failed and why (so future sessions don't repeat them)
   - Bottlenecks discovered in the CSV parser (e.g. "delimiter scanning is memory-bound, not compute-bound")
   - Architecture insights about how the parser works internally
   - Useful papers or techniques found during research
   
   Memory persists across sessions — even if context compacts or a new session starts, these notes survive. Reference memory at the start of each session and during re-anchoring to recall past discoveries.

10. **Decision**:
    - Improved beyond noise floor, or simpler at equal perf → **keep**
    - Within noise floor, regressed, or more complex at equal perf → **discard** with `git reset --hard HEAD~1`

11. **Clean up**: `rm -f build.log test.log run.log`

12. **Discipline checks** (before next iteration):
    - **Circuit breaker**: 3 consecutive `discard` or `crash` results? You're going down the wrong path. STOP. Re-read the source code from disk, re-read results.tsv, do fresh web research, and pick a fundamentally different approach.
    - **Force diversity**: 3+ variations of the same technique (e.g. different thread block sizes, different shared memory tile configs)? You're stuck in a local optimum. Try a completely different algorithm. The biggest gains come from algorithmic changes, not parameter sweeps.
    - **Re-anchor every 5 experiments**: Re-read results.tsv end-to-end, check `/memory` for past discoveries, re-state your objective, summarize what worked and what failed, only THEN propose your next hypothesis. Long sessions cause context rot — memory is your hedge against it.
    - **Idea backlog low** (fewer than 2-3 ideas)? Respawn 2-3 researcher agents with the current results.tsv so they know what's been tried and search in new directions.

13. **Repeat.**

**Timeout**: Build > 30 min, test > 10 min, or benchmark > 30 min → kill and treat as failure.

**Crashes**: If something dumb and easy to fix (typo, missing include), fix and re-run. If fundamentally broken, log `crash`, discard, move on.

**Output hygiene**: Always redirect to log files. Build output can be thousands of lines:
```bash
# Correct
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1
tail -n 20 build.log

# Wrong — floods context
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
```

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

**Unlimited.** Cost is not a concern — the only goal is maximizing performance improvements. Always use Opus 4.6 (1M context) at maximum effort for all work including subagents. Do unlimited web searches. Context auto-compacts as needed — don't worry about window limits.

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

## Quick start

To start a new experiment run, just say:

> Optimize the CSV parser for maximum throughput.

Or use the command: `/project:experiment csv`
