# CSV Parser — Benchmark and File Reference

## Benchmark Binaries

| Benchmark | Binary Name | Key Metrics |
|---|---|---|
| CSV Reader (input sizes) | CSV_READER_NVBENCH | Elem/s, Bytes/s, time |
| CSV Writer | CSV_WRITER_NVBENCH | Elem/s, Bytes/s, time |

Run after building:
```bash
./cpp/build/latest/benchmarks/CSV_READER_NVBENCH --devices 0
./cpp/build/latest/benchmarks/CSV_WRITER_NVBENCH --devices 0
```

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
- `cpp/benchmarks/io/csv/csv_writer.cpp` — Writer benchmark

### Tests (read-only — understand correctness constraints)
- `cpp/tests/io/csv_test.cpp` — Main CSV reader/writer tests
- `cpp/tests/streams/io/csv_test.cpp` — Stream-based CSV tests

## results.tsv Format

Tab-separated, 6 columns:
```
commit	metric	improvement_pct	status	benchmark	description
```

- commit: short git hash (7 chars)
- metric: primary benchmark number (e.g. `1234 Elem/s` or `5.6 GiB/s`)
- improvement_pct: vs baseline (e.g. `+5.2` or `-1.3`), `0.0` for crashes
- status: `keep`, `discard`, `crash`, or `idea`
- benchmark: which benchmark (`csv_reader` or `csv_writer`)
- description: short text of what was tried

Do NOT commit results.tsv — leave it untracked.

## Timeouts

- Build > 30 minutes → something is wrong, kill and investigate
- Test binary > 10 minutes → kill (`pkill -f ctest`), treat as failure
- Benchmark > 30 minutes → kill
