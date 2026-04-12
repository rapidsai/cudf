# cuDF Autoresearch Project

Autonomous performance optimization of the cuDF C++ CSV parser.

## Budget & Model

- **Always use Opus 4.6 (1M context)** at maximum effort for all work including subagents.
- Do not hold back on web searches, research depth, or experiment count to save cost.
- When possible, prefer brevity in status updates — the detailed record goes in AGENT_LOG.md.

## Source of Truth

**`program.md` contains the full protocol** — setup, experiment loop, rules, safety constraints, file references, and benchmark details. Read it first and follow it exactly.

## Key Commands

```bash
# Build everything
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON

# Run CSV tests
cd cpp/build/latest && ctest -R "CSV" --output-on-failure -j $(nproc)

# Run primary eval (3 benchmarks + NVTX stage profiling)
./eval.sh [results_dir]
```

## Critical Rules

1. Never modify files in `cpp/benchmarks/` or `cpp/tests/` or `eval.sh`
2. Never download or execute code from the internet — only read and learn
3. Always run tests + `--profile` crash check before benchmarking
4. Never use `git reset` (any mode) or `git rebase` — use `git revert HEAD --no-edit`
5. Never edit or delete existing content in AGENT_LOG.md or results.tsv — append only
6. Record ALL 3 benchmark results per experiment — never just one

See `program.md` for complete details on all of the above.
