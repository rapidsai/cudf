---
name: improve-compilation-cudf
description: Use this skill to measure and improve C++ compilation times in cudf.
---

## Check if we are in devcontainer

Detect cudf devcontainer usage by checking if the username is `coder`. If not, ignore everything else in this file and skip.

## Goal

Iteratively reduce cudf C++ compilation times by measuring, identifying the slowest translation units, planning a targeted improvement, executing it, and validating the result. Repeat until satisfied.

## Key principle: prefer splits for parallelism

It is always better to have several files with shorter compilation times that compile in parallel, even if the sum of those split times is comparable to the compile time of the larger file before the split. Wall-clock build time is what matters, not per-file totals.

## Process

This is an iterative cycle. After each committed improvement, return to step 1 to re-baseline.

### 1. Baseline build (clean, once)

Run the baseline script **once at the start** to establish a full timing baseline. It configures cmake with sccache disabled, tests and benchmarks enabled, native arch only, `--threads=1` for nvcc, then does a clean build and generates reports. Tests and benchmarks must stay enabled so CMake settings don't change between measurement and validation (changing flags invalidates the build and forces full rebuilds). **The build may take up to 4 hours.** When invoking via the Bash tool, set the timeout to at least 14400000 ms (4 hours).

```bash
.agents/skills/improve-compilation-cudf/baseline.sh
```

Or override the job count (default is `nproc --ignore=2`):

```bash
.agents/skills/improve-compilation-cudf/baseline.sh 12
```

The script prints the top 20 slowest translation units and saves reports (CSV, HTML, ninja_log) to `compilation_reports/`. It outputs a `TAG=` line at the end — save this as `BASELINE_TAG` for later comparison.

### 2. Identify the slowest translation units

The baseline script prints the top 20 automatically. To re-examine:

```bash
sort -t',' -k1 -nr "compilation_reports/${BASELINE_TAG}.csv" | head -20
```

The HTML report includes a color-coded Gantt chart (red = >5min, yellow = 2-5min, green = 1s-2min).

### 3. Write an improvement plan

Create a plan file:

```bash
touch "compilation_reports/${TIMESTAMP}_${COMMIT}_plan.md"
```

The plan must:
- Target a specific object file from the top of the slowest list.
- Diagnose why that file is slow (heavy template instantiation, large header fan-out, excessive includes).
- Propose a specific, narrow fix using one of the strategies listed below.

### 4. Execute the plan

Make the code changes. If splitting a file, add the new source files to the relevant `CMakeLists.txt` target.

### 5. Validate: incremental rebuild and timing

After making changes, **do NOT re-run the full baseline script**. Instead, do an incremental rebuild and measure only the affected translation units.

#### 5a. Reconfigure if CMakeLists.txt changed

If you added or removed source files from `CMakeLists.txt`, reconfigure first:

```bash
ninja -C cpp/build/latest reconfigure
```

#### 5b. Delete only the affected object files

Remove the `.o` files for the translation units you changed or split so ninja will rebuild just those:

```bash
rm -f cpp/build/latest/CMakeFiles/cudf.dir/src/path/to/changed_file.cu.o
rm -f cpp/build/latest/CMakeFiles/cudf.dir/src/path/to/new_split_file.cu.o
```

#### 5c. Rebuild and capture per-file times

Rebuild using ninja. Ninja appends to `.ninja_log`, so the new entries will reflect only the rebuilt files:

```bash
ninja -C cpp/build/latest cudf -j$(nproc --ignore=2) 2>&1
```

Also ensure the stream usage test libraries are built (required by ctest's `LD_PRELOAD` wrapper):

```bash
ninja -C cpp/build/latest \
  cudf_identify_stream_usage_mode_cudf \
  cudf_identify_stream_usage_mode_testing \
  -j$(nproc --ignore=2) 2>&1
```

Then generate a comparison report against the baseline:

```bash
TAG_NEW="$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)"
cp cpp/build/latest/.ninja_log "compilation_reports/${TAG_NEW}.ninja_log"

python cpp/scripts/sort_ninja_log.py cpp/build/latest/.ninja_log --fmt csv \
  > "compilation_reports/${TAG_NEW}.csv"

python cpp/scripts/sort_ninja_log.py cpp/build/latest/.ninja_log --fmt html \
  --cmp_log "compilation_reports/${BASELINE_TAG}.ninja_log" \
  > "compilation_reports/${BASELINE_TAG}_vs_${TAG_NEW}.html"
```

#### 5d. Verify improvement

Check the affected files compiled faster (or that the new split files each compile faster than the original single file). Verify no other file regressed significantly:

```bash
# Show times for the affected files
grep -E 'changed_file|new_split_file' "compilation_reports/${TAG_NEW}.csv"

# Show top 20 to confirm no regressions
sort -t',' -k1 -nr "compilation_reports/${TAG_NEW}.csv" | head -20
```

For splits: confirm each new file is faster than the original, and that they compile in parallel (wall-clock improvement matters, not sum of individual times).

### 6. Run tests

After verifying compilation improvement, run the test suite to confirm correctness. The baseline script already configures with `BUILD_TESTS=ON`, so tests are always built alongside the library — no reconfiguration needed.

**IMPORTANT:** Do NOT use the devcontainer wrapper scripts (`build-cudf-cpp`, `test-cudf-cpp`, `configure-cudf-cpp`) for this process. They may silently override CMake flags (sccache launchers, GPU architectures, build type, etc.) and invalidate the controlled measurement environment. Use raw ninja/ctest invocations instead.

For targeted changes, run the specific test suites that exercise the modified code:

```bash
ctest --test-dir cpp/build/latest -R <RELEVANT_TEST_PATTERN> --output-on-failure
```

For broader changes, run the full test suite:

```bash
ctest --test-dir cpp/build/latest -j10 --output-on-failure
```

If tests fail, fix the issue and repeat from step 5.

### 7. Run benchmarks before/after and compare

**IMPORTANT:** Capture "before" benchmarks **before** making code changes, so you don't have to revert and rebuild to get a baseline.

#### 7a. Before making code changes — run and save "before" benchmarks

Identify the relevant nvbench executable(s) for the code you're about to change. Build them if needed, then run:

```bash
# Example for groupby changes:
ninja -C cpp/build/latest GROUPBY_NVBENCH -j$(nproc --ignore=2)
cpp/build/latest/benchmarks/GROUPBY_NVBENCH --json compilation_reports/bench_before.json
```

#### 7b. After making code changes — rebuild and run "after" benchmarks

After your incremental rebuild (step 5), rebuild the benchmark executable (it links against libcudf) and run:

```bash
ninja -C cpp/build/latest GROUPBY_NVBENCH -j$(nproc --ignore=2)
cpp/build/latest/benchmarks/GROUPBY_NVBENCH --json compilation_reports/bench_after.json
```

#### 7c. Compare results programmatically

Parse both JSON files and compute the ratio (after/before) for each benchmark configuration. Flag any regression >5%:

```python
import json, statistics

before = {s["name"]: s for s in json.load(open("compilation_reports/bench_before.json"))["benchmarks"]}
after  = {s["name"]: s for s in json.load(open("compilation_reports/bench_after.json"))["benchmarks"]}

ratios = []
for name in sorted(set(before) & set(after)):
    b = before[name]["average"]["center"]
    a = after[name]["average"]["center"]
    ratio = a / b if b else float("inf")
    ratios.append(ratio)
    if ratio > 1.05:
        print(f"REGRESSION {name}: {b:.3f} -> {a:.3f} ({ratio:.4f}x)")

print(f"\nConfigurations compared: {len(ratios)}")
print(f"Mean ratio: {statistics.mean(ratios):.4f}")
print(f"Median ratio: {statistics.median(ratios):.4f}")
```

A mean/median ratio near 1.0 with no individual regressions >5% confirms no runtime performance impact. If regressions are found, investigate before committing.

### 8. Pre-commit checks

Before committing, stage all changed files and run `pre-commit run --all-files`. Pre-commit hooks (clang-format, copyright year, cmake-format, etc.) may modify files. After it completes:

1. Re-stage any files modified by the hooks (`git add` the affected files).
2. Do a quick incremental rebuild (`ninja -C cpp/build/latest cudf -j$(nproc --ignore=2)`) to confirm the formatting changes didn't break compilation.
3. Then commit — the pre-commit hooks will run again as part of `git commit` and should all pass.

```bash
git add <changed-files>
pre-commit run --all-files
# If hooks modified files:
git add <re-modified-files>
ninja -C cpp/build/latest cudf -j$(nproc --ignore=2)
git commit -m "..."
```

### 9. Iterate

After committing, return to step 1 only if you need a fresh full baseline. Otherwise, the incremental ninja log from step 5 can serve as the new baseline for the next iteration:

```bash
BASELINE_TAG="${TAG_NEW}"
```

---

## Strategies

Proven techniques from past cudf compilation improvements, ordered roughly by frequency of use and impact.

### Split large translation units
Split a slow `.cu`/`.cpp` file into multiple smaller files that compile in parallel. This is the single most-used strategy in cudf history. Examples: `rolling.cu` (300s to 60s parallel, #6512), `scan.cu` (11% overall build speedup, 17% smaller libcudf.so, #8183), hash join (split into 19 files). When splitting, add the new source files to the relevant `CMakeLists.txt` target. Even if the sum of split file times equals the original, wall-clock time improves because they compile in parallel.

### Reduce template instantiations
Replace type-dispatched templates with type-erased alternatives or runtime parameters. The **indexalator** (`cudf/detail/indexalator.cuh`) normalizes all integer index types into a single iterator, eliminating per-type instantiations. Converting a template parameter to a runtime parameter (e.g., `has_nulls` bool, `join_kind` enum) can halve compile time with no runtime cost. Example: `thrust::tabulate` to `thrust::transform` + `counting_iterator` reduced `group_rank_scan.cu` from 8m45s to 1m14s (#21793).

A constrained dispatcher that only handles the subset of template parameters actually used by a code path can dramatically cut instantiations. For example, if a file only uses 11 of 36 aggregation kinds, a constrained `aggregation_dispatcher` eliminates ~70% of instantiations.

### Replace heavy `.cuh` includes with lighter `.hpp` includes
Device-code headers (`.cuh`) pull in CUDA/Thrust/CUB internals. When the including file only needs declarations, switch to the corresponding `.hpp`. Example: replacing `gather.cuh` with `gather.hpp` in 22 files significantly reduced object sizes (#9299).

### Move definitions from headers to source files
Non-template function definitions in widely-included headers force recompilation of every includer. Move them to `.cpp`/`.cu` files. Examples: `table_view.hpp` inline functions (#14535), `ast_expression` definitions (#19250).

### Forward declarations
Replace `#include` with a forward declaration when only a pointer or reference to the type is needed. Examples: `cudf::io::data_sink`, `cudaStream_t`, groupby `sort_helper`.

### Include cleanup (IWYU)
Run include-what-you-use to find and remove unnecessary includes. A single IWYU pass removed 643 includes across 260 files (#17170). IWYU is integrated into CI and can be run locally:
```bash
cmake --build cpp/build/latest --target iwyu 2>&1 | tee iwyu_output.txt
```

### PIMPL (pointer to implementation)
Hide implementation details behind a `unique_ptr<impl>` in the public header. Consumers only see the forward-declared impl type, insulating them from changes to the implementation and its heavy includes. Used for `hash_join` and `sort_merge_join` (#21349).

### Explicit template instantiation with extern declarations
Instantiate templates in a single `.cu` file and use `extern template` in the header so other TUs don't redundantly instantiate them. Used in the groupby hash split (#17089).

### `__noinline__` for debug builds
Mark secondary type-dispatching device functions as `__noinline__` to prevent nvcc from inlining deeply nested template expansions in debug builds. Used for row-operators that were causing >1.5hr debug builds (#21197).

### Object libraries for shared utilities
Compile shared test/benchmark utilities once as a CMake object library instead of recompiling them per-test. Applied to `cudftestutil` (#18131).

---

## Reference

| Resource | Path |
|----------|------|
| Baseline build script | `.agents/skills/improve-compilation-cudf/baseline.sh` |
| Ninja log analyzer | `cpp/scripts/sort_ninja_log.py` |
| Build script (with `--build_metrics`) | `build.sh` |
| Developer guide | `cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md` |
| Build skill | `.agents/skills/build-test-cudf/SKILL.md` |
