# cuDF Autoresearch Project

This project is set up for autonomous performance optimization of the cuDF C++ CSV parser.

## Budget & Model

- **Unlimited budget** — cost is not a concern. The only goal is maximizing performance improvements.
- **Always use Opus 4.6 (1M context)** — use the most capable model at maximum effort for all work.
- **Auto-compact context** — let context auto-compact as needed. Don't worry about context window limits.
- All subagents (researcher, etc.) should also use Opus 4.6 1M unless there's a strong reason for a faster model.
- Do not hold back on web searches, research depth, or experiment count to save cost.

## What This Is

An experiment harness where Claude autonomously optimizes the cuDF C++ CSV parser for maximum GPU performance through systematic research-driven experimentation. See `program.md` for the full protocol — it is the source of truth.

## Key Commands

```bash
# Build everything
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON

# Run CSV tests
cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc)

# Run CSV benchmarks
./cpp/build/latest/benchmarks/CSV_READER_NVBENCH --devices 0
./cpp/build/latest/benchmarks/CSV_WRITER_NVBENCH --devices 0

# Extract benchmark metrics
grep -E "Elem/s|Bytes/s|GlobalMem BW|BWUtil|time" run.log
```

## Project Structure

- `cpp/src/io/csv/` — primary CSV source files (start here)
- `cpp/src/` and `cpp/include/` — full editable zone (CSV has dependencies across the source tree)
- `cpp/benchmarks/io/csv/` — CSV benchmarks (READ-ONLY)
- `cpp/tests/io/csv_test.cpp` — CSV tests (READ-ONLY)
- `program.md` — the full autoresearch protocol (source of truth)
- `results.tsv` — experiment log (untracked)

## Working with Tools and CLIs

Before writing custom logic to work around a perceived limitation, check whether the tool already handles it. Run `--help` or read the docs first. Don't wrap, guard, or re-implement behavior that the tool provides natively.

## Critical Rules

1. Never modify files in `cpp/benchmarks/` or `cpp/tests/`
2. Never download or execute code from the internet — only read and learn
3. Always redirect build/test/benchmark output to log files to avoid context flooding
4. Always run tests before benchmarking — correctness is non-negotiable
5. Clean up log files after each experiment cycle
