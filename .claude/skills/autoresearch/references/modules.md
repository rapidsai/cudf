# CSV Parser — Benchmark and File Reference

## Benchmark Binaries

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

### Running benchmarks

```bash
# Primary eval (every experiment) — runs 3 benchmarks + NVTX profiling
./eval.sh results/<experiment_tag>

# Holistic view (every 3 experiments) — runs all reader benchmarks
RESULTS_DIR="results/<experiment_tag>_full" && mkdir -p "$RESULTS_DIR"
for f in cpp/build/latest/benchmarks/CSV_READER_*; do
  name=$(basename "$f")
  "$f" --timeout 5 --json "$RESULTS_DIR/$name.json"
done
```

## NVTX Stage Profiling

`eval.sh` includes NVTX profiling on the TAXI/256MB realistic benchmark. Instrumented stages:

- `csv::load_data_and_gather_row_offsets`
- `csv::select_data_and_row_offsets`
- `csv::infer_column_types`
- `csv::determine_column_types`
- `csv::decode_data`

Output: `results/<tag>/nvtx_stages.txt`

## Files to Read

### Primary target (CSV parser source)
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

## results.tsv Format

Tab-separated, 7 columns. **Each experiment produces 3 rows** — one per primary benchmark.

```
exp	commit	metric	improvement_pct	status	benchmark	description
```

- exp: sequential experiment number (0 = baseline, 1, 2, 3, ...)
- commit: short git hash (7 chars)
- metric: benchmark throughput (e.g. `1234 bytes/s` or `5.6 GiB/s`)
- improvement_pct: vs baseline (e.g. `+5.2` or `-1.3`), `0.0` for crashes
- status: `keep`, `discard`, `crash`, or `idea`
- benchmark: `realistic`, `type_inference`, or `quoting`
- description: short text of what was tried

Do NOT commit results.tsv — leave it untracked.

## AGENT_LOG.md Format

Append-only narrative log. One section per experiment:

```markdown
## Experiment N: <short title>

**Hypothesis**: <what you changed and why>
**Result**: <keep/discard/crash> — <one-line summary of numbers>

### What worked
- <bullet points>

### What didn't
- <bullet points>

### What I learned
- <insights about the parser, GPU behavior, or algorithm>

### Next direction
- <what the research head plans to try next and why>
```

Do NOT commit AGENT_LOG.md — leave it untracked.

## Timeouts

- Build > 30 minutes → something is wrong, kill and investigate
- Test binary > 10 minutes → kill (`pkill -f ctest`), treat as failure
- Benchmark > 30 minutes → kill
