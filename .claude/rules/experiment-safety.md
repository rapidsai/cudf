---
paths:
  - "cpp/**"
---

# Experiment Safety Rules

## Read-Only Zones (Non-Negotiable)

These directories define ground truth. Modifying them invalidates all experiment results.

- **`cpp/benchmarks/**`** — Benchmarks define what is measured. Never modify.
- **`cpp/tests/**`** — Tests define correctness. If tests fail, the code is wrong, not the tests.

## Editable Zone

- **`cpp/src/**`** and **`cpp/include/**`** — These directories may be edited. The primary target is `cpp/src/io/csv/` but the CSV parser has dependencies across IO utilities, common infrastructure, and type dispatching. Everything is fair game: algorithms, data structures, kernel implementations, memory access patterns, thread configurations, warp-level optimizations, shared memory usage.

## API Contract

Public API function signatures should be preserved — do not change existing public function signatures or remove public functions/types. Adding new internal/detail helpers and new overloads that don't break existing callers is fine.

## Build System

- Do NOT modify `CMakeLists.txt` unless strictly necessary for new source files you've added
- Do NOT install new packages or add dependencies beyond what's in `pyproject.toml` / `CMakeLists.txt`

## Output Hygiene

Always redirect command output to log files. Build output can be thousands of lines and will flood context:

```bash
# Correct
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON > build.log 2>&1
tail -n 20 build.log

# Wrong — floods context
build-cudf-cpp -j0 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
```

Clean up logs after each experiment: `rm -f build.log test.log run.log`
