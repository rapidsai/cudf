# cuDF Autoresearch Project

This project is set up for autonomous performance optimization of the cuDF C++ CSV parser.

## Budget & Model

- **Unlimited budget** — cost is not a concern. The only goal is maximizing performance improvements.
- **Always use Opus 4.6 (1M context)** — use the most capable model at maximum effort for all work.
- **Auto-compact context** — let context auto-compact as needed. Don't worry about context window limits.
- All subagents (researcher, etc.) should also use Opus 4.6 1M unless there's a strong reason for a faster model.
- Do not hold back on web searches, research depth, or experiment count to save cost.

## Zero Human Intervention

This is a **fully autonomous** experiment loop. **Never ask the user any questions.** Never pause for confirmation, approval, or "say the word" prompts. Make all decisions yourself:

- **Prioritize experiments yourself** — rank by expected impact vs risk, then execute in that order.
- **Handle failures yourself** — if a build fails, fix it. If a test fails, revert and move on. If an idea doesn't pan out, pick the next one.
- **Never wait for human input** — there is no human watching. If you stop and ask a question, the experiment stalls forever.
- **Use your judgment** — you have all the context you need. Research, decide, implement, measure, log, repeat.

The only human action is starting the session. Everything after that is on you.

## What This Is

An experiment harness where Claude autonomously optimizes the cuDF C++ CSV parser for maximum GPU performance through systematic research-driven experimentation. See `program.md` for the full protocol — it is the source of truth.

## Key Commands

```bash
# Build everything
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON

# Run CSV tests
cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc)

# PRIMARY benchmark (multi-datatype — the optimization target)
./cpp/build/latest/benchmarks/CSV_READER_NVBENCH -b csv_read_io --devices 0

# Diagnostic: single-type benchmarks (for investigating per-type regressions)
./cpp/build/latest/benchmarks/CSV_READER_NVBENCH -b csv_read_data_type --devices 0

# Writer benchmark
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
- `results.tsv` — experiment log with numbered experiments (untracked)
- `AGENT_LOG.md` — append-only experiment journal: hypothesis, result, learnings (untracked)
- `MCP_SETUP_NEEDED.md` — MCP/plugin setup notes requiring user action (created on demand)

## Critical Rules

1. Never modify files in `cpp/benchmarks/` or `cpp/tests/`
2. Never download or execute code from the internet — only read and learn
3. Always redirect build/test/benchmark output to log files to avoid context flooding
4. Always run tests before benchmarking — correctness is non-negotiable
5. Clean up log files after each experiment cycle
