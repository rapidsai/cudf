---
name: perf-compare-cudf
description: Use when the user invokes /perf-compare-cudf, asks to benchmark a cudf branch or PR against main, or wants a performance comparison for libcudf changes.
---

Use this skill when the user invokes `/perf-compare-cudf` with either:
- **nothing / "current branch"** (default) — benchmark the currently checked-out branch, or
- **a cudf PR link / number** (e.g. `https://github.com/rapidsai/cudf/pull/12345`) — check out that PR first.

# Goal
Run a curated subset of nvbench benchmarks on the PR branch, then on a branch in sync with `<cudf-remote>/main`, and produce a markdown report comparing the two with significant changes highlighted.

Note the cudf GH repository link: https://github.com/rapidsai/cudf
`<cudf-remote>` = the git remote pointing at cudf GH repo (often `upstream`). Detect it with `git remote -v`.

---

## Workflow checklist

Copy and track progress:

```
- [ ] 0. Confirm devcontainer + env (/build-test-cudf skill)
- [ ] 1. Determine target + which case applies (A: target is current branch / B: target is a different PR branch)
- [ ] 2. Get onto the PR side per the case (Case A: stay put, keep local changes; Case B: record start branch, stash, checkout clean PR branch)
- [ ] 3. Decide which benchmarks to run (infer from diff; confirm with user)
- [ ] 4. Decide axis coverage (full vs skim); confirm with user
- [ ] 5. Pick an idle GPU (nvidia-smi)
- [ ] 6. Build libcudf with -DBUILD_BENCHMARKS=ON on PR side
- [ ] 7. Run benchmarks on PR side -> results/pr/
- [ ] 8. Stash now (Case A only — preserve local changes), then switch to a branch in sync with latest <cudf-remote>/main
- [ ] 9. Rebuild libcudf with benchmarks on main
- [ ] 10. Run the SAME benchmark invocations on main -> results/main/
- [ ] 11. Generate comparison report
- [ ] 12. Restore: return to the starting branch, unstash; clean up any temp branch created
```

---

## Step 0: Devcontainer + build environment

Read and follow `/build-test-cudf` (skill at `.agents/skills/build-test-cudf/SKILL.md`) to:
- Confirm we are in a cudf devcontainer (username `coder`). If not, stop.
- Get and follow instructions to build (and troubleshoot builds) libcudf with specified CMake options

## Step 1: Determine target and which case applies

Record the starting branch (`git branch --show-current`). Then classify into one of two cases — this decides how local changes and stashing are handled:

- **Case A — target is the current branch.** Applies when no PR was given, OR a PR was given and the current branch already is the PR's branch (tracks it). **Keep local uncommitted changes as-is; do NOT stash yet.** We build & benchmark the PR side first, then stash only when leaving for main.
- **Case B — target is a PR on a different branch.** Applies when a PR was given and the current branch is NOT the PR's branch.

For a PR, resolve its head branch (to compare against current and to check out in Case B). Use `gh` if available:

```bash
gh pr view <PR_NUMBER> --repo rapidsai/cudf --json headRefName,headRepository,headRepositoryOwner,baseRefName,title
```

If `gh` auth is unavailable, fetch the PR ref directly:

```bash
git fetch <cudf-remote> pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
```

## Step 2: Get onto the PR side

- **Case A:** stay on the current branch. Do not stash and do not checkout — local changes stay applied through the PR-side build and benchmark run.
- **Case B:** the PR side must be clean, so:
  1. Record the starting branch and stash any uncommitted changes, **recording exactly what was stashed** so it can be restored at the very end:

     ```bash
     git stash push -m "perf-compare-cudf: wip" -- <paths...>   # or `git stash push -m ...` for everything
     git stash list
     ```
  2. Check out the PR branch: reuse an existing local branch tracking it; otherwise check out the fetched ref (`git checkout pr-<PR_NUMBER>`). `git checkout <headRefName>` works only when a local/remote-tracking branch of that name already exists (e.g. same-repo PRs).

## Step 3: Decide which benchmarks to run

Infer candidate benchmarks from the diff vs `<cudf-remote>/main`:

```bash
git fetch <cudf-remote> main
git diff --name-only <cudf-remote>/main...HEAD
```

Map changed source areas to benchmark binaries. Benchmarks are built at `cpp/build/latest/benchmarks`. List all with `ls cpp/build/latest/benchmarks` and inspect a binary's benchmarks/axes with `<BENCH> --list`.

Examples of mapping:
- Parquet writer changes → `PARQUET_WRITER_NVBENCH`
- Parquet reader / cuIO source changes → `PARQUET_READER_NVBENCH`, `PARQUET_READER_COMPRESSED_NVBENCH`, `PARQUET_MULTITHREAD_READER_NVBENCH`, `HYBRID_SCAN_*_NVBENCH`, etc.
- Join changes → `JOIN_NVBENCH`; sort → `SORT_NVBENCH`; strings → `STRINGS_NVBENCH`.

**Always confirm the benchmark list with the user** (use AskQuestion if available). The user may name specific benchmarks, give a hint, or ask to run all of them.

## Step 4: Decide axis coverage

Each nvbench benchmark has axes with many values; full cross-product is large. Ask the user (AskQuestion) whether to:
- **Skim** (default, fast): pick a few representative values per axis — smallest, largest, and a couple in the middle.
- **Full**: run all configurations.

Inspect axes per binary:

```bash
cpp/build/latest/benchmarks/<BENCH> --list      # shows axes + values
cpp/build/latest/benchmarks/<BENCH> --help-axes  # axis spec syntax
```

Override axes with `-a name=[v1,v2,...]` (and `-b <bench_name>` to scope). **Record the exact axis values chosen** and use the identical invocation on both branches.

## Step 5: Pick an idle GPU

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

Choose one idle GPU (0% util, ~0 MiB used). Run on a single device only. Pass `CUDA_VISIBLE_DEVICES=<idx>` and also `-d 0` to every nvbench invocation (after masking, the chosen device is index 0).

## Step 6: Build libcudf (PR side)

Use the `/build-test-cudf` skill to configure libcudf with `-DBUILD_BENCHMARKS=ON` CMake option and build. **Do not miss** this CMake option as libcudf benchmarks will not build otherwise. In Case A this builds with the local uncommitted changes applied (intended). In case of errors, use the same skill to troubleshoot or clean, reconfigure **with the CMake option** and rebuild as needed.

## Step 7: Run benchmarks on PR side

Write CSV + log per benchmark binary into a results dir (suggest `.agents/benchmark-results/<run-id>/pr/`). For each benchmark binary:

```bash
CUDA_VISIBLE_DEVICES=<idx> cpp/build/latest/benchmarks/<BENCH> -d 0 \
  -b <bench_name> -a <axis=...> ... \
  --csv <out>/pr/<BENCH>.csv 2>&1 | tee <out>/pr/<BENCH>.log
```

Notes:
- nvbench may emit a benign segfault at the very end of a suite — **ignore end-of-suite segfaults**.
- If a config throws an exception, note it and exclude it from the comparison, but verify the **same** behavior occurs on both branches.

## Step 8: Stash (Case A) and switch to a branch in sync with rapidsai/cudf main

Now that PR-side data is collected, leave for the main side. The "main" side must reflect `<cudf-remote>/main` (rapidsai/cudf), not a stale fork main, and must NOT carry the PR's changes.

1. **Case A only:** the local changes are still applied — stash them now so main is clean, **recording exactly what was stashed**:

   ```bash
   git stash push -m "perf-compare-cudf: wip" -- <paths...>   # or `git stash push -m ...` for everything
   git stash list
   ```

   (Case B already stashed in Step 2 and is on a clean PR branch.)
2. `git fetch <cudf-remote> main`
3. Decide the main-side ref:
   - If local `main` already tracks `<cudf-remote>/main` and is up to date → `git checkout main`.
   - If local `main` tracks a fork → pull `<cudf-remote>/main` into it, **or** create a temp branch from the remote: `git checkout -b _bench_main <cudf-remote>/main` (delete it in Step 12).
4. Do NOT apply the stash on main — main must stay clean of PR changes so the comparison is meaningful.

## Step 9: Rebuild on main

Follow the instructions in **Step 6: Build libcudf (PR side)** to properly configure and rebuild on the "main" branch. Note that the "main" branch must be in a clean state.

## Step 10: Run the SAME benchmarks on main

Use the identical `-b`/`-a` invocations and the same GPU, writing to `<out>/main/`.

## Step 11: Generate the comparison report

Use the helper script (reads matching CSVs and emits markdown):

```bash
python .agents/skills/perf-compare-cudf/scripts/compare.py \
  --pr <out>/pr --main <out>/main --report <out>/COMPARISON.md
```

The script matches rows by `(benchmark, axis-values)`, computes `Δ = (PR - main) / main` on GPU time, and flags **significant** changes where `|Δ| >= 5%` AND `|Δ| >` max(noise of either side). It prints significant rows to stdout.

**Re-run any flagged config once** to confirm it is not noise (especially sub-millisecond benches with high nvbench noise). If the rerun matches within noise, replace that CSV row's run and regenerate.

See the report template at the bottom.

## Step 12: Restore state

- Return to the starting branch recorded in Step 1 (Case A: the original/current branch; Case B: the branch we started on before checking out the PR).
- Pop the stash to restore the user's uncommitted changes (both cases stashed something: Case A in Step 8, Case B in Step 2).
- Delete any temp branch created in Step 8 (e.g. `git branch -D _bench_main`).
- Confirm `git status` matches the pre-run state.

---

## Report template

The generated `COMPARISON.md` (and your chat summary) should include:

```markdown
# Benchmark Comparison: <cudf-remote>/main vs PR (`<branch or PR>`)

- GPU Time in ms. Δ = (PR - main) / main. Negative = PR faster.
- Significant: |Δ| >= 5% AND larger than max(noise) of either side.
- Hardware: <GPU name + index from nvbench/nvidia-smi>, driver/CUDA if available.
- Branches: PR `<sha>` vs main `<sha>`. Axis coverage: <skim|full> (list values used).

## Summary
| Benchmark Suite | # Configs | # Significant |
| ... |

## Top N by |Δ|
| Suite / bench | axes | main (ms) | PR (ms) | Δ | noise(m/p) |
| ... |

## Per-suite tables
（one table per benchmark, axes as columns, with FASTER/SLOWER flags）

## Notes
- Exceptions excluded (same on both branches): ...
- End-of-suite segfaults ignored.
- Files generated: list of CSV/log paths + this report + compare.py
```

Always end the chat summary with: the headline (regression / improvement / within-noise), the hardware, the axis coverage, and the list of generated files.
