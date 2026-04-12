---
paths:
  - "cpp/**"
---

# Experiment Safety Rules

## Read-Only Zones (Non-Negotiable)

These directories define ground truth. Modifying them invalidates all experiment results.

- **`cpp/benchmarks/**`** — Benchmarks define what is measured. Never modify.
- **`cpp/tests/**`** — Tests define correctness. If tests fail, the code is wrong, not the tests.
- **`eval.sh`** — The eval script is fixed. Never modify.

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

## Reverting Failed Experiments

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
