# autoresearch — cuDF CSV Parser Performance Optimization

Autonomous LLM-driven optimization of the cuDF C++ CSV parser for maximum GPU throughput through systematic research-driven experimentation.

## Run Configuration

- **Base branch**: `origin/dev_autoresearch_v3`
- **Run tag pattern**: `<monthday>-csv` (e.g. `apr12-csv`)

## Research Seeds

Starting directions for this run. NOT prescriptive — validate each idea against profiling data before implementing.

- (Add per-run research seeds here before starting a run)

### Known Dead Ends (do NOT retry)

- (Add known failed approaches here before starting a run)

---

## The Goal

**Optimize the CSV reader for maximum throughput on mixed-type workloads.** The primary target is `CSV_READER_REALISTIC_NVBENCH` — it parses multiple data types (ints, floats, timestamps, strings) in a single `read_csv` call. The other two primary benchmarks (`CSV_READER_TYPE_INFERENCE_NVBENCH`, `CSV_READER_QUOTING_NVBENCH`) provide coverage for type inference overhead and quoting/FSM paths.

All 3 primary benchmarks are run via `./eval.sh` after every experiment. **Every experiment must report results for ALL 3 benchmarks.** A change that speeds up one but regresses another is not a win.

**Representative configurations**: Each primary benchmark has multiple configs (type profiles, sizes, quoting levels). For keep/discard decisions and results.tsv, use the **mixed-type representative config** from each benchmark:
- `realistic` → **TAXI / 256MB** (14 mixed-type columns — ints, floats, timestamps, strings)
- `type_inference` → **MIXED / with_inference / 256MB** (mixed types with inference enabled — the real-world default)
- `quoting` → **QUOTE_25_PCT / 256MB** (realistic quoting density — 25% quoted string columns)

Single-type configs (ALL_INTEGRAL, ALL_FLOAT, ALL_STRING, QUOTE_0_PCT, QUOTE_100_PCT) exist in the benchmark output but are **diagnostic only** — they help explain WHY a change helped or hurt, but must NOT drive optimization direction. Do not optimize for single-type throughput.

**Focus: CSV reader only.** The reader (`reader_impl.cu`, `csv_gpu.cu`) is the performance-critical path. Writer benchmarks are out of scope.

**Algorithm-first principle**: Prefer algorithmic improvements over GPU-specific hyperparameter tuning. Changes should produce runtime improvements across all GPUs and architectures — current, newer, and similar-class hardware. Concretely:

- **Always prefer**: Better parsing algorithms, reduced memory passes, smarter data structures, reduced work, better parallelization strategies
- **Accept when justified**: Architecture-aware memory access patterns (coalescing, avoiding bank conflicts) — these transfer across GPU generations
- **Avoid unless no algorithmic alternative exists**: Tuning thread block dimensions, warp-specific tile sizes, shared memory capacity knobs, occupancy-chasing launch configs

If you find yourself tweaking numeric parameters rather than changing the structure of the computation, you are in the wrong optimization layer.

**Simplicity criterion**: All else being equal, simpler is better. Removing something and getting equal or better results is a great outcome.

- Equal perf + simpler code = **keep**
- 1% gain + 50 lines of hacks = **probably not worth it**
- 1% gain from deleting code = **definitely keep**
- 0% gain + much simpler code = **keep**

### Benchmark traps to avoid

**Micro-benchmark tunnel vision.** Single-type benchmarks measure one data type in isolation — essentially the best case. This can lead to writing specialized per-type kernels instead of optimizing the common mixed-type path, or tunnel-visioning on one technique applied type by type. **The REALISTIC benchmark (TAXI/LOGS/ANALYTICS) is ground truth** — if you find yourself applying the same technique to each type one by one, STOP. You're optimizing micro-benchmarks, not real-world performance.

**Synthetic vs real-world gap.** NVBench uses clean, pre-typed, uniform data. Real CSV files have mixed types, type inference overhead, quoted fields with special characters, variable-length strings, NULL/NA values, headers, comments, and BOM markers. A technique that wins on synthetic data may not help on messy real data.

---

## Scope

**Editable**: `cpp/src/` and `cpp/include/` — everything is fair game: parsing algorithms, GPU kernels, memory access patterns, shared infrastructure.

**Read-only (non-negotiable)**: `cpp/benchmarks/`, `cpp/tests/`, `eval.sh`. Benchmarks define what is measured. Tests define correctness — if tests fail, the code is wrong. Never modify these.

**API contract**: Preserve existing public function signatures. Adding new internal/detail helpers is fine.

**Build system**: Do NOT modify `CMakeLists.txt` unless adding a new source file. No new packages or dependencies.

**Web research**: Read and learn freely (papers, CUDA docs, CUB/Thrust API docs, competing implementations). Do NOT download or execute external code (`curl | bash`, `wget`, `git clone`, `pip install`, etc.). Understand techniques and reimplement from scratch.

**MCP / Plugins**: May install trusted MCP servers or CLI plugins on the fly for analysis, profiling, or research. If a plugin requires user authentication, note requirements in `SETUP_REQUIRED.md` and continue. Do not install unverified or code-executing plugins.

---

## Setup

1. **Pick a run tag** (e.g. `apr12-csv`). Branch `autoresearch/<tag>` must not already exist.
2. **Create branch**:
   ```bash
   git fetch origin
   git rev-parse --verify <base_branch> || { echo "ERROR: base branch does not exist. STOP."; exit 1; }
   git checkout -b autoresearch/<tag> <base_branch>
   ```
   If the base branch does not exist, **STOP and tell the user**.
3. **Read source code** — all files listed in File Reference below, starting with `cpp/src/io/csv/`.
4. **Read Research Seeds and Known Dead Ends** at the top of this file.
5. **Deep research phase** — spawn 2-3 researcher agents in parallel:
   - **Researcher 1**: CSV parsing algorithms — GPU-accelerated CSV/text parsing, parallel field detection, single-pass vs multi-pass, work-reduction techniques
   - **Researcher 2**: Data structure and memory-pass optimization — reducing passes, intermediate representations, stage fusion, scan-based algorithms (e.g. CUB decoupled lookback)
   - **Researcher 3**: Competing implementations — how DuckDB, Apache Arrow CSV, ParaText, etc. parse CSV (parsing strategy, pass count, type-handling approach)

   Pass Known Dead Ends to each. Read source code yourself while they run. Merge results into a ranked backlog.
6. **Build baseline**: `build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1`
   If OOM, retry with `-j$(nproc)` or lower.
7. **Run baseline tests**: `cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1`
8. **Establish noise floor** — run `eval.sh` 3 times without code changes:
   ```bash
   ./eval.sh results/baseline_run1
   ./eval.sh results/baseline_run2
   ./eval.sh results/baseline_run3
   ```
   Record average AND variance. Future improvements must exceed this noise floor.
9. **Initialize** results.tsv (header + baseline exp 0), benchmarks.tsv (header + baseline exp 0 with all configs), and AGENT_LOG.md (run header).
10. **Begin the loop.**

---

## The Experiment Loop

LOOP FOREVER:

### 1. Research & Hypothesize

Act as **research head** before every experiment:
- Read the latest `nvtx_stages.txt` — which stage dominates? (`load_data`, `decode_data`, `infer_column_types`, etc.)
- Read the last few entries of `AGENT_LOG.md` — what worked, what failed, what patterns are emerging?
- Spawn **1-2 small focused researcher agents** with a specific question (not broad surveys). Examples:
  - "The decode_data stage is 70% of runtime. Find papers on GPU-parallel type conversion for mixed int/float/string columns."
  - "Warp divergence is high in csv_gpu.cu:decode_field. Find techniques for reorganizing work so threads process similar column types together."
- Review results.tsv to avoid repeating failed approaches.

Then state your hypothesis explicitly: what you're changing and why, what metric you expect to improve and by how much, and what could go wrong.

**Researcher cadence**: 2-3 broad researchers at start of run, 1-2 focused researchers per experiment, 2-3 deep researchers after stall (3+ experiments without improvement).

### 2. Implement

Modify files in `cpp/src/` and `cpp/include/`. One idea per experiment — don't mix unrelated changes.

### 3. Commit

Stage only code files you changed. Do NOT use `git add -A` or `git add .`.
```bash
git add cpp/src/ cpp/include/
git commit -m "<description>"
```

### 4. Build

```bash
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1
```
Check: `tail -n 20 build.log`. If it fails, max 3 fix attempts, then abandon.

**While the build runs**: spawn researcher agents for the NEXT experiment; read source code or profiling data. Do NOT edit files during a build. When the build completes, proceed to Test immediately.

### 5. Test

Both gates must pass before ANY benchmarking:

**a) Unit tests:**
```bash
cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc) > ../../test.log 2>&1
```

**b) Benchmark crash check:**
```bash
for f in cpp/build/latest/benchmarks/CSV_*; do
  "$f" --profile --timeout 10 > /tmp/bench_check.log 2>&1 || echo "CRASH: $(basename $f)"
  grep -qi "error\|illegal\|misaligned\|fault" /tmp/bench_check.log && echo "CUDA ERROR in $(basename $f)"
done
```

If either gate fails → immediately revert, log as `crash`, move on. Do NOT benchmark broken code — a CUDA error poisons the driver state.

### 6. Benchmark

**a) Primary eval** (drives keep/discard decision):
```bash
./eval.sh results/<experiment_tag>
```

**b) Full reader benchmarks** (run EVERY experiment for comprehensive logging):
```bash
RESULTS_DIR="results/<experiment_tag>"
for f in cpp/build/latest/benchmarks/CSV_READER_*; do
  name=$(basename "$f")
  "$f" --timeout 5 --json "$RESULTS_DIR/$name.json"
done
```

After both complete, append all results to `benchmarks.tsv` (see Logging section). This produces a complete record of every benchmark config for every experiment — essential for tracking regressions across the full benchmark surface and for post-run analysis.

### 7. Evaluate & Record

- Is the improvement larger than the noise floor? If not, it's noise.
- Is it >20% from a minor change? Re-run eval.sh twice more to confirm.
- Does it hold across ALL 3 primary benchmarks, or only one?
- Check `nvtx_stages.txt` — did the expected stage speed up?

Record in all 3 log files:
- **results.tsv**: 3 rows (one per primary benchmark, using representative mixed-type config throughput)
- **benchmarks.tsv**: one row per config across ALL reader benchmarks (complete snapshot)
- **AGENT_LOG.md**: append experiment section

After significant discoveries, save insights to `/memory`.

### 8. Decide

- Improved beyond noise floor, or simpler at equal perf → **keep**
- Within noise floor, regressed, or more complex at equal perf → **discard** with `git revert HEAD --no-edit`

Clean up: `rm -f build.log test.log`

### 9. Repeat

Check discipline rules below before the next iteration. Every 5th experiment, print a compact progress summary: best result so far, total experiments, current direction.

---

## Logging

There are 3 log files. All are **untracked, append-only** — never edit existing entries, never change column format mid-run.

### results.tsv — agent decision log

Tab-separated. 3 rows per experiment (one per primary benchmark), using the **representative mixed-type config** for each benchmark's metric:
- `realistic` → TAXI / 256MB throughput
- `type_inference` → MIXED / with_inference / 256MB throughput
- `quoting` → QUOTE_25_PCT / 256MB throughput

Header and 7 columns:

```
exp	commit	metric	improvement_pct	status	benchmark	description
```

| Column | Format |
|---|---|
| exp | Sequential number (0 = baseline, 1, 2, 3...) |
| commit | Short git hash (7 chars) |
| metric | Throughput from the representative config (e.g. `5.2 GiB/s`) |
| improvement_pct | vs baseline (e.g. `+5.2` or `-1.3`), `0.0` for crashes |
| status | `keep`, `discard`, `crash`, or `idea` |
| benchmark | `realistic`, `type_inference`, or `quoting` |
| description | What was tried. Extra context (single-type deltas, NVTX details, re-run notes) goes here |

### benchmarks.tsv — comprehensive benchmark log

Tab-separated. Logs **every config of every reader benchmark** for every experiment. This is the complete record for post-run analysis — the user will script over this to see how all numbers changed across experiments.

Format is **fixed for the entire run**. Header and 6 columns:

```
exp	commit	status	binary	config	throughput_GiBs
```

| Column | Format |
|---|---|
| exp | Sequential experiment number (matches results.tsv) |
| commit | Short git hash (7 chars) |
| status | `keep`, `discard`, `crash` (matches results.tsv decision) |
| binary | Benchmark binary name (e.g. `CSV_READER_REALISTIC_NVBENCH`) |
| config | Config string describing the axes (e.g. `TAXI/256MB`, `MIXED/with_inference/256MB/8cols`, `QUOTE_25_PCT/256MB`) |
| throughput_GiBs | Throughput as a number (e.g. `5.23`) |

Extract throughput for each config from the NVBench JSON `bytes_per_second` field, convert to GiB/s. One row per benchmark config per experiment. For crashes, log 0 rows (no benchmark data).

Initialize with the header row, then append baseline (exp 0) rows for all configs. Every subsequent experiment appends a complete set of rows for all configs.

### AGENT_LOG.md — narrative log

Append-only. After every experiment:

```markdown
## Experiment N: <short title>
**Hypothesis**: <what and why>
**Result**: <keep/discard/crash> — <numbers>
### What worked / What didn't / What I learned / Next direction
```

Initialize with: `# Agent Log — <run tag>`

### NVTX stage profiling

`eval.sh` profiles 5 stages via `nsys` on TAXI/256MB:
- `csv::load_data_and_gather_row_offsets` — H2D transfer + row offset detection
- `csv::select_data_and_row_offsets` — byte/row range selection
- `csv::infer_column_types` — type inference (skipped with explicit dtypes)
- `csv::determine_column_types` — final column type determination
- `csv::decode_data` — main GPU parsing/decoding pass

---

## Safety & Recovery

### Reverting

**NEVER use `git reset` (any mode), `git rebase`, or `git checkout .`** These rewrite history and can destroy untracked files (AGENT_LOG.md, results.tsv).

To discard: `git revert HEAD --no-edit`. This creates a forward-only inverse commit touching only code files.

### Output hygiene

Always redirect to log files. Build output floods context:
```bash
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1
tail -n 20 build.log
```

### Crash recovery

| Failure | Response |
|---|---|
| Build fails | Max 3 fix attempts, then revert and abandon |
| Tests fail | Immediately revert, log as `crash` |
| Benchmark crashes / CUDA error | Immediately revert — CUDA errors poison driver state |
| Build > 30 min | Kill and investigate |
| Test > 10 min | Kill (`pkill -f ctest`), treat as failure |
| Benchmark > 30 min | Kill |
| Surprising result (>20% gain) | Re-run eval.sh twice more to confirm |

---

## Discipline Rules

Quality over quantity. Every rejected experiment wastes 10-30 minutes. A well-reasoned hypothesis beats five speculative shots.

### Rule 1 — One idea per experiment
Related changes (e.g. a new algorithm + the memory layout it requires) belong together. Unrelated ideas must be separate experiments. Experiments are numbered sequentially: Exp1, Exp2, Exp3. Researcher ideas are NOT experiment numbers — an idea becomes an experiment only when you commit code. Fix-up commits keep the same number.

### Rule 2 — Verify surprising results
If a result is >20% gain from a minor change, re-run the benchmark twice more. Only trust reproducible numbers.

### Rule 3 — Stall detection → deep research
If 3+ experiments show no improvement, STOP and enter deep research: re-read source from disk, re-read results.tsv and AGENT_LOG.md end-to-end, re-read NVTX stages, spawn 2-3 deep researcher agents. Require a high-confidence idea before the next experiment.

### Rule 4 — Force diversity (algorithm-first)
If 3+ variations of the same technique (different block sizes, tile configs, unroll factors), STOP. Try a completely different algorithmic approach. The biggest gains come from fewer passes, fused stages, reduced total work — not GPU-specific parameter sweeps.

### Rule 5 — Re-anchor every 5 experiments
Long sessions cause context rot. Every 5th experiment: re-read this file from disk, re-read results.tsv and AGENT_LOG.md, check `/memory` for past discoveries, re-state your objective, summarize what worked and failed, save new insights to memory, then propose next hypothesis.

### Rule 6 — No scope creep
Do not "also optimize" a neighboring module. Do not refactor outside the critical path. Do not add utilities unless directly required for the current experiment. Off-topic ideas go in results.tsv as `idea` status rows.

### Rule 7 — Don't strip working code outside the hot path
If existing code handles edge cases or special types the benchmark doesn't exercise, leave it alone. Removing functionality to simplify the benchmark path is metric gaming. Tests verify correctness for all cases.

### When stuck

Do NOT start tweaking GPU-specific parameters. Instead:
1. More web searches — new papers, adjacent GPU workloads (JSON parsing, text processing, regex on GPU)
2. Re-read source code looking for missed bottlenecks
3. Combine two previous near-miss ideas that each showed partial improvement
4. Try a fundamentally different approach
5. Spawn researcher agents — always pass results.tsv so they know what's been tried

---

## Reference

### Primary benchmarks (run via `eval.sh` — drive keep/discard decisions)

| Benchmark | Binary | Representative Config | What It Measures |
|---|---|---|---|
| Realistic mixed-type | CSV_READER_REALISTIC_NVBENCH | **TAXI / 256MB** | TAXI (14 mixed cols), LOGS (6 cols), ANALYTICS (8 cols) at 256/512/1024 MB |
| Type inference | CSV_READER_TYPE_INFERENCE_NVBENCH | **MIXED / with_inference / 256MB** | With/without inference; single-type configs are diagnostic only |
| Quoting density | CSV_READER_QUOTING_NVBENCH | **QUOTE_25_PCT / 256MB** | 0%, 25%, 100% quoted columns at 64/256 MB |

### Additional reader benchmarks (run every experiment for comprehensive logging)

| Benchmark | Binary | What It Measures |
|---|---|---|
| Original reader (input sizes) | CSV_READER_NVBENCH | Per-type and IO source variations (diagnostic — do not optimize for single-type results) |
| Scale (large data) | CSV_READER_SCALE_NVBENCH | Mixed-type from 256 MB to 4 GB |

### File reference

**Primary target (start here):**
`cpp/src/io/csv/` — `reader_impl.cu`, `csv_gpu.cu`, `csv_gpu.hpp`, `csv_common.hpp`, `durations.cu`, `durations.hpp`, `datetime.cuh`, `writer_impl.cu`

**Editable zone:** `cpp/src/`, `cpp/include/`

**Public API (preserve interface):** `cpp/include/cudf/io/csv.hpp`, `cpp/include/cudf/io/detail/csv.hpp`

**Benchmarks (read-only):** `cpp/benchmarks/io/csv/csv_read_realistic.cpp` (primary), `csv_read_type_inference.cpp` (primary), `csv_read_quoting.cpp` (primary), `csv_reader_input.cpp`, `csv_reader_options.cpp`, `csv_read_scale.cpp`, `csv_writer.cpp`, `csv_write_scale.cpp`

**Tests (read-only):** `cpp/tests/io/csv_test.cpp`, `cpp/tests/streams/io/csv_test.cpp`

### Optimization areas (by impact tier)

**Algorithmic (highest — explore FIRST):**
- **Pass reduction**: Fuse or overlap the multi-kernel pipeline (row detection → field detection → type conversion) to reduce memory traffic
- **Work reduction**: Skip unnecessary computation (type inference when types are known, redundant scans, common-case short-circuits)
- **Better parallelization**: Reorganize work so threads process similar types together, reducing warp divergence
- **Efficient scan algorithms**: CUB decoupled lookback and similar state-of-the-art scan algorithms
- **Stage fusion**: Combine delimiter detection + field extraction + type conversion to avoid materializing intermediates
- **Pipelining**: Overlap H2D transfer with compute for large files
- **Host-side overhead**: Reduce memory allocation, column construction, metadata processing

**Structural (moderate — follows fundamental hardware design, transfers across generations):**
- Parallel delimiter/newline scanning, vectorized type conversion algorithms, coalesced memory access patterns, parallel quote handling (FSM-based), row/column work decomposition for variable-length rows

**Type-specific (lower — only after algorithmic approaches exhausted):**
- Hot-path data type conversion (branchless, lookup tables), duration/datetime parsing improvements

**Hardware-specific tuning (LAST RESORT):**
- Thread block dimensions, shared memory tile sizes, occupancy knobs, unroll factors — fragile gains tied to one GPU model

---

## Autonomy

**NEVER STOP.** Do not pause to ask the human if you should continue. The human may be asleep. Continue indefinitely until manually stopped. Each experiment takes ~10-30 minutes, so you can run 2-6 per hour, 15-50 overnight.

**Budget**: Opus 4.6 (1M context) at maximum effort for all work including subagents. Unlimited web searches. Context auto-compacts as needed.

**Memory**: Use `/memory` to persist discoveries across sessions. Check it at session start and during re-anchoring. Save significant findings: successful optimizations (with numbers), failed approaches (with reasons), bottleneck analysis, useful papers. Memory complements results.tsv — the TSV tracks metrics, memory tracks the *why*.

---

## Quick start

> Optimize the CSV parser for maximum throughput.

Or: `/project:experiment csv`
