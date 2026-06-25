---
name: perf-compare-cudf
description: Benchmark a cuDF branch, WIP changes, or a PR against the `main` branch
---

Use this skill when the user asks to compare libcudf benchmark performance for:
- **the current branch or WIP changes** against `rapidsai/cudf` `main`.
- **a cudf PR link or number** against `rapidsai/cudf` `main`.

# Goal

Run the same selected libcudf NVBench benchmarks on the target and then on `rapidsai/cudf` `main`, then report meaningful differences.

`<cudf-remote>` is the git remote for `https://github.com/rapidsai/cudf` (often `upstream`). Detect it with `git remote -v`.

## Prerequisites

- **`gh` CLI** authenticated — run `gh auth status`. If not authenticated, guide the user to run:
   ```bash
   gh auth login
   ```
  The token needs `repo` scope. Do **not** run `gh auth token` from within the agent.
- Ensure we are in cudf devcontainer. Otherwise ensure that the CUDA, compilers, and cudf build helpers are available.

## 1. Prepare

- Record the starting branch, `git status --short`, and the exact target (current WIP or cudf PR).
- Before switching branches, stash unrelated user changes and record the stash name. If the target is the current WIP, keep changes applied for the target run, then stash them before switching to main.
- Check out the PR with:
  ```bash
  gh pr checkout <PR_NUMBER> --repo rapidsai/cudf
  ```

## 2. Choose Benchmarks

- Fetch current main with `git fetch <cudf-remote> main`.
- Infer candidates from `git diff --name-only <cudf-remote>/main...HEAD`.
- Build outputs live under `cpp/build/latest/benchmarks/*_NVBENCH`.
- Inspect a binary with:
  ```bash
  cpp/build/latest/benchmarks/<BENCH> --list
  cpp/build/latest/benchmarks/<BENCH> --help-axes
  ```
- Confirm benchmark binaries and axis coverage with the user. Use a small, representative axis subset by default; use full coverage only when requested or necessary.
- Record exact `-b` and `-a` options. Reuse them unchanged on both branches.

## 3. Build each side

On both target and main, force CMake reconfiguration to enable benchmarks via `-DBUILD_BENCHMARKS=ON` before building:

```bash
configure-cudf-cpp -DBUILD_BENCHMARKS=ON
build-cudf-cpp
```

Use `/build-test-cudf` skill for configure, build instructions and troubleshooting.

## 4. Run each side

- Pick an idle GPU with `nvidia-smi`. Re-check every time before running anything (target or main run); if the same GPU is no longer idle, pick another one, wait or ask before continuing.
- Run one masked device only: `CUDA_VISIBLE_DEVICES=<idx>` and `-d 0`.
- Write matching JSON and log files, for example:
  ```bash
  CUDA_VISIBLE_DEVICES=<idx> cpp/build/latest/benchmarks/<BENCH> -d 0 \
    -b <bench_name> -a <axis=...> ... \
    --json <out>/pr/<BENCH>.json 2>&1 | tee <out>/pr/<BENCH>.log
  ```
- To switch to main, stash target WIP if needed, then use a clean branch:
  ```bash
  git fetch <cudf-remote> main
  git checkout -B _bench_main <cudf-remote>/main
  ```
  Do not apply target changes on `_bench_main`.
- Repeat the identical command on main, writing to `<out>/main/`.
- If nvbench emits an end-of-suite segfault after writing results, note it and continue. If a config throws, verify whether both branches behave the same.

## 5. Compare
Use NVBench's comparison script from the build tree:

```bash
NVBENCH_SCRIPTS=cpp/build/latest/_deps/nvbench-src/python/scripts
test -f "$NVBENCH_SCRIPTS/nvbench_compare.py" || \
  NVBENCH_SCRIPTS=cpp/build/latest/_deps/nvbench-src/scripts
PYTHONPATH="$NVBENCH_SCRIPTS" python "$NVBENCH_SCRIPTS/nvbench_compare.py" \
  --threshold-diff 0.05 --no-color <out>/main <out>/pr \
  | tee <out>/COMPARISON.md
```

The first path is the reference (`main`), the second is the comparison (`pr`). Re-run surprising failures once, especially small or noisy configs.

## 6. Restore and report

- Return to the starting branch, pop any stash you created, delete temporary branches, and confirm `git status` matches the starting state.
- Summarize chat with the headline result (regression, improvement, or within noise), relevant metrics, hardware used, branch SHAs, axis coverage, and generated files.
- Use this report shape for `COMPARISON.md`, adapting the metric columns to the benchmark. GPU time is always useful, but other metrics such as output file size, throughput, compression ratio, or memory usage are also of interest when they change significantly in target vs main.

  ```markdown
  # Benchmark Comparison: <cudf-remote>/main vs target (`WIP` or `PR`)

  - Primary metric(s): <GPU time, throughput, output size, memory, etc.>
  - Δ = (target - main) / main. Interpret direction per metric.
  - Significant timing deltas: |Δ| >= 5% AND larger than max(noise) of either side.
  - Hardware: <GPU name + index from nvbench/nvidia-smi>, driver/CUDA if available.
              <CPU name + cores from `lscpu`>, model, architecture, if available.
  - Branches: target `<sha>` vs main `<sha>`. Axis coverage: <skim|full> (list values used).

  ## Summary
  | Benchmark Suite | Primary metric | # Configs | # Meaningful Changes |
  | ... |

  ## Top N Changes
  | Suite / bench | axes | metric | main | target | Δ | noise, if timing |
  | ... |

  ## Per-suite tables
  (one table per benchmark, axes as columns, include all relevant metrics)

  ## Notes
  - Exceptions excluded (same on both branches): ...
  - End-of-suite segfaults ignored.
  - Files generated: list of JSON/log paths + this report
  ```
