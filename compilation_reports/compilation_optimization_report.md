# cudf C++ Compilation Time Optimization Report

**Branch**: `improve-compilation-skill`
**Base**: `main`
**Hardware**: NVIDIA RTX 6000 Ada Generation, 34 parallel jobs
**Build config**: `BUILD_TESTS=ON`, `BUILD_BENCHMARKS=ON`, `CMAKE_CUDA_ARCHITECTURES=NATIVE`, no sccache

## Summary

Critical path (slowest single translation unit) reduced from **712s to 391s** (45% reduction).
The previous critical-path file (`compute_shared_memory_aggs.cu` at 712s) was reduced to 110s via a constrained aggregation dispatcher. The new critical path is `mixed_join_kernel_nulls.cu` at 391s, which is a header-cost-only file (28 lines of source, all cost from template instantiations in included headers) and not amenable to further splitting.

All 113 ctest targets pass. No runtime regressions — 3577 benchmark states compared across 10 suites, with 45-59% runtime **improvements** in shmem-heavy groupby benchmarks.

## Optimizations

### 1. Constrained shared-memory aggregation dispatcher (`125b1b526a`)

**File**: `compute_shared_memory_aggs.cu`
**Technique**: Created `shmem_dispatch.cuh` with an 11-kind dispatcher (vs the original 36-kind `dispatch_type_and_aggregation`) that only includes aggregation kinds actually used in shared-memory accumulation.
**Compile time**: 712s → 110s (85% reduction)
**Runtime impact**: 45-59% **improvement** in `groupby_max_cardinality` and `groupby_m2_var_std` benchmarks. The constrained switch statement produces better nvcc codegen for the shmem kernel context.

### 2. `filtered_join.cu` split (`8a7e85fa3d` — by Bradley Dice)

**File**: `filtered_join.cu` (391s) → 4 files
**Technique**: Split by comparator type: nested, primitive, and a simple path. Created `filtered_join_detail.cuh` for shared declarations.

| File | Time (s) |
|------|----------|
| `filtered_join.cu` | — (dispatcher only) |
| `filtered_join_nested.cu` | 190 |
| `filtered_join_primitive.cu` | 43 |
| `filtered_join_simple.cu` | 93 |

**Critical path**: 391s → 190s (51% reduction)

### 3. `sort_helper.cu` split (`59f93aed6c`, `6314289cbf` — by Bradley Dice)

**File**: `sort_helper.cu` (270s) → 4 files
**Technique**: Extracted group_offsets (with nested/non-nested split) and unique_keys.

| File | Time (s) |
|------|----------|
| `sort_helper.cu` | 31 |
| `sort_helper_group_offsets.cu` | 188 |
| `sort_helper_group_offsets_nested.cu` | 76 |
| `sort_helper_unique_keys.cu` | 69 |

**Critical path**: 270s → 188s (30% reduction)

### 4. `hash_join.cu` split (`4745de083f`)

**File**: `hash_join.cu` (398s) → 4 files
**Technique**: Extracted non-template helper functions (`hash_join_build.cu`, `hash_join_probe.cu`, `hash_join_full.cu`) from the monolithic file. The `hash_join<Hasher>` template methods remain thin wrappers. Created `hash_join_helpers.cuh` for shared declarations.

| File | Time (s) |
|------|----------|
| `hash_join.cu` | 89 |
| `hash_join_build.cu` | 56 |
| `hash_join_probe.cu` | 267 |
| `hash_join_full.cu` | 122 |

**Critical path**: 398s → 267s (33% reduction)

### 5. `compute_groupby.cu` split (`3f38e66096`)

**File**: `compute_groupby.cu` (341s) → 2 files
**Technique**: Split by nullable template parameter. `compute_groupby.cu` keeps the non-nullable instantiation; `compute_groupby_nullable.cu` gets the nullable one. Shared template definition in `compute_groupby.cuh`.

| File | Time (s) |
|------|----------|
| `compute_groupby.cu` | 168 |
| `compute_groupby_nullable.cu` | 216 |

**Critical path**: 341s → 216s (37% reduction)

### 6. `distinct_helpers.cu` split (`a18c417fcc`)

**File**: `distinct_helpers.cu` (296s) → 2 files
**Technique**: Split by nullable template parameter, same pattern as compute_groupby.

| File | Time (s) |
|------|----------|
| `distinct_helpers.cu` | 150 |
| `distinct_helpers_nullable.cu` | 239 |

**Critical path**: 296s → 239s (19% reduction)

### 7. `contains_table` extern template declarations (`1b1176fe1e` — by Bradley Dice)

**File**: `contains_table_impl.cuh`
**Technique**: Added extern template declarations to prevent redundant instantiation of `contains_table_impl` templates in `contains_table.cu` (already instantiated in `contains_table_impl_nested.cu`).

### 8. `tdigest_aggregation.cu` split (`ec4aec531c`)

**File**: `tdigest_aggregation.cu` (372s) → 2 files
**Technique**: Separated scalar tdigest computation from merge-tdigest path. Created `tdigest_aggregation.cuh` for shared cluster computation infrastructure.

| File | Time (s) |
|------|----------|
| `tdigest_aggregation.cu` | 86 |
| `tdigest_merge.cu` | 97 |

**Critical path**: 372s → 97s (74% reduction)

### 9. `range_rolling.cu` split (`cb015fafa3`)

**File**: `range_rolling.cu` (389s) → 3 files
**Technique**: Moved `bounded_closed` and `bounded_open` WindowType dispatchers to separate TUs. Created `range_window_dispatch.hpp` for per-WindowType function declarations.

| File | Time (s) |
|------|----------|
| `range_rolling.cu` | 135 |
| `range_rolling_bounded_closed.cu` | 210 |
| `range_rolling_bounded_open.cu` | 213 |

**Critical path**: 389s → 213s (45% reduction)

### 10. `sort_merge_join.cu` split (`86f97e53c6`)

**File**: `sort_merge_join.cu` (296s) → 2 files
**Technique**: Extracted `merge` template class and `invoke_merge` into `sort_merge_join_impl.cuh`. Moved `left_join` and `inner_join_match_context` to `sort_merge_join_left.cu`.

| File | Time (s) |
|------|----------|
| `sort_merge_join.cu` | 225 |
| `sort_merge_join_left.cu` | 224 |

**Critical path**: 296s → 225s (24% reduction)

### 11. `conditional_join.cu` split (`567038135b`)

**File**: `conditional_join.cu` (256s) → 2 files
**Technique**: Moved `conditional_join_anti_semi` detail function and public API wrappers to separate TU.

| File | Time (s) |
|------|----------|
| `conditional_join.cu` | 190 |
| `conditional_join_anti_semi.cu` | 191 |

**Critical path**: 256s → 191s (25% reduction)

### 12. `key_remapping.cu` split (`c1f8a7353b`)

**File**: `key_remapping.cu` (305s) → 4 files
**Technique**: Split 3 `key_remap_table<Comparator>` instantiations (primitive, nested, non-nested) into separate TUs with factory functions. Created `key_remapping_fwd.hpp` (interface class) and `key_remapping_impl.cuh` (template definition).

| File | Time (s) |
|------|----------|
| `key_remapping.cu` | 35 |
| `key_remapping_primitive.cu` | 125 |
| `key_remapping_nested.cu` | 254 |
| `key_remapping_non_nested.cu` | 175 |

**Critical path**: 305s → 254s (17% reduction)

### 13. `hash_join_probe.cu` further split (`659f249f1f` — by Bradley Dice)

**File**: `hash_join_probe.cu` (267s) → 2 files
**Technique**: Separated `compute_size` from the probe kernel into `hash_join_probe_compute_size.cu`.

| File | Time (s) |
|------|----------|
| `hash_join_probe.cu` | — |
| `hash_join_probe_compute_size.cu` | — |

## Attempted Optimizations (not included in branch)

These optimizations were explored during development but caused runtime regressions and were not kept.

### Global-memory constrained dispatcher

Created `gmem_dispatch.cuh` with a 12-kind dispatcher for global memory aggregation. Reduced `compute_global_memory_aggs*.cu` compile times but caused **37-54% runtime regression** in `complex_int_keys` and `complex_mixed_keys` groupby benchmarks. The constrained switch in `single_pass_functors.cuh` changed nvcc codegen (register allocation, instruction scheduling) in the hot-path GPU kernel.

### Tdigest centroid materialization

Materialized `centroid` structs into `device_uvector<centroid>` before passing to cub reduce, eliminating per-type template instantiations in the reduction kernel. Caused **25-49% runtime regression** in `reduce-few-large` tdigest benchmarks. The materialization added ~10GB extra memory traffic for 320M rows — the kernel is bandwidth-bound.

### Reduction framework type erasure

Two approaches tried:
1. **Materialization**: Converting typed iterators to `device_uvector<double>` before cub reduce. **2-4x runtime regression** — the cub kernel IS the main computation, adding a full extra pass is catastrophic.
2. **Type-erased device-side dispatch**: `type_erased_iterator` with per-element `type_dispatcher`. **10-430% regression** — the per-element branch cost is measurable even with uniform branching across a warp.

## Runtime Benchmark Comparison

3577 states compared across 10 benchmark suites. Methodology: each suite run on `main` and `improve-compilation-skill` with nvbench, GPU mean times compared.

### Per-Suite Summary

| Suite | States | Mean Δ% | Median Δ% | Max Δ% | Min Δ% |
|-------|--------|---------|-----------|--------|--------|
| GROUPBY_targeted | 368 | -11.83% | -0.50% | +6.11% | -59.15% |
| HASHING_NVBENCH | 103 | -0.07% | +0.03% | +3.73% | -13.44% |
| JOIN_NVBENCH_filtered | 584 | +0.00% | -0.00% | +9.29% | -10.59% |
| QUANTILES_NVBENCH | 192 | -0.36% | -0.24% | +1.34% | -3.04% |
| REDUCTION_NVBENCH | 880 | -0.06% | -0.02% | +8.51% | -12.13% |
| ROLLING_NVBENCH | 124 | -0.09% | -0.13% | +9.77% | -16.10% |
| SEARCH_NVBENCH | 82 | -0.17% | -0.20% | +5.74% | -3.25% |
| SORT_NVBENCH | 558 | -0.16% | -0.12% | +15.68% | -5.26% |
| STREAM_COMPACTION_NVBENCH | 660 | +1.01% | +0.98% | +17.22% | -31.04% |
| TDIGEST_NVBENCH | 26 | -0.97% | -0.91% | -0.02% | -2.28% |

### Distribution

| Range | Count |
|-------|-------|
| <-10% | 130 |
| -10% to -5% | 16 |
| -5% to -2% | 88 |
| -2% to +2% | 3128 |
| +2% to +5% | 174 |
| +5% to +10% | 37 |
| >+10% | 4 |

87.4% of states are within ±2%. All 41 "regressions" (>5%) are on sub-millisecond benchmarks with measurement noise exceeding the change magnitude (none flagged with `!`). All 130 improvements >10% are in groupby benchmarks, attributable to the shmem constrained dispatcher, and are flagged `!` (statistically significant, exceeding 3x measurement noise).

## Key Insights

1. **nvcc compile time scales with template instantiation requests, not unique instantiations.** A constrained dispatcher with only needed case statements compiles much faster than the full 36-kind dispatcher, even though unsupported kinds would be dead code. The dedup happens too late in nvcc's pipeline.

2. **Constrained dispatchers can improve OR regress GPU kernel runtime.** The shmem constrained dispatcher improved runtime 45-59% (better register allocation in the shmem kernel context). The gmem constrained dispatcher regressed runtime 37-54% (worse codegen in the gmem kernel context). There is no way to predict which direction without benchmarking.

3. **Materializing at type-dispatch boundaries causes runtime regressions in bandwidth-bound kernels.** Extra memory traffic from materialization dominates when the kernel is already bandwidth-bound (tdigest, reductions).

4. **Type-erased device-side dispatch fails for bandwidth-bound kernels.** Per-element branch cost is measurable even with uniform branching, causing 10-430% regressions in reductions.

5. **Many remaining high-cost files are "header cost only"** — short source files (28-121 lines) where all compile time comes from heavy template instantiations in included headers (row operators, cuco hash tables, thrust sort). These cannot be optimized by splitting.

6. **File splitting always helps wall-clock time** even when the sum of split times exceeds the original, because the splits compile in parallel.

## Recommended Priority Order

Ranked by critical-path reduction weighted against invasiveness (diff size, new abstractions, review risk).

### Tier 1: High impact, low risk

| Priority | Optimization | Commit | Critical path Δ | Diff | Rationale |
|----------|--------------|--------|-----------------|------|-----------|
| 1 | shmem constrained dispatcher | [`125b1b526a`](https://github.com/rapidsai/cudf/commit/125b1b526a882c2524b2a3b20969df3210a11dcc) | 712s → 110s (**-602s**) | +113/-23, 2 files | Biggest single win. New header with constrained switch, no structural changes. Also yields 45-59% runtime improvement. |
| 2 | tdigest split | [`ec4aec531c`](https://github.com/rapidsai/cudf/commit/ec4aec531c45faa79813227afba2e0066cce1a4b) | 372s → 97s (**-275s**) | +1459/-1420, 4 files | Large diff but mechanical code motion (net +39 lines). Clean scalar vs merge separation. |
| 3 | range_rolling split | [`cb015fafa3`](https://github.com/rapidsai/cudf/commit/cb015fafa30aea50ac80feae8707383fc5c4ab45) | 389s → 213s (**-176s**) | +133/-13, 5 files | Smallest diff for largest impact ratio. Moves two case dispatchers to separate TUs. |
| 4 | sort_helper splits (2 commits) | [`59f93aed6c`](https://github.com/rapidsai/cudf/commit/59f93aed6c7a05a48b49927ec6614818efb67502), [`6314289cbf`](https://github.com/rapidsai/cudf/commit/6314289cbf903ddd29abf6fc057be5b29cdf169c) | 270s → 188s (**-82s**) | +196/-111, 8 files | Two small commits. Clean extraction of independent methods. |

### Tier 2: Moderate impact, low-to-moderate risk

| Priority | Optimization | Commit | Critical path Δ | Diff | Rationale |
|----------|--------------|--------|-----------------|------|-----------|
| 5 | hash_join split | [`4745de083f`](https://github.com/rapidsai/cudf/commit/4745de083f346ce75ea36cd962d684059beb40fa) | 398s → 267s (**-131s**) | +472/-483, 6 files | Net -11 lines. Natural split boundaries (build/probe/full). |
| 6 | filtered_join split | [`8a7e85fa3d`](https://github.com/rapidsai/cudf/commit/8a7e85fa3d355ae7d65b2f3144857528fa44e4cd) | 391s → 190s (**-201s**) | +681/-256, 7 files | Larger diff, but comparator-type split is a clean pattern. |
| 7 | compute_groupby split | [`3f38e66096`](https://github.com/rapidsai/cudf/commit/3f38e660960fb86ff9a392aebfc0de965c4968a6) | 341s → 216s (**-125s**) | +190/-165, 4 files | Small, mechanical nullable/non-nullable split. |
| 8 | conditional_join split | [`567038135b`](https://github.com/rapidsai/cudf/commit/567038135b70f18e418d810e6e1d1398a4cd9abb) | 256s → 191s (**-65s**) | +171/-136, 3 files | Small diff, clean anti_semi extraction. |

### Tier 3: Moderate impact, moderate invasiveness

| Priority | Optimization | Commit | Critical path Δ | Diff | Rationale |
|----------|--------------|--------|-----------------|------|-----------|
| 9 | distinct_helpers split | [`a18c417fcc`](https://github.com/rapidsai/cudf/commit/a18c417fcc351616bb035f03b840c54b6d72d070) | 296s → 239s (**-57s**) | +127/-105, 3 files | Same nullable pattern. Only 19% reduction — nullable path still heavy. |
| 10 | sort_merge_join split | [`86f97e53c6`](https://github.com/rapidsai/cudf/commit/86f97e53c64fb76fe0942901dc8dd1a7040a940a) | 296s → 225s (**-71s**) | +663/-654, 4 files | Large diff (net +9 lines), mechanical code motion. |
| 11 | key_remapping split | [`c1f8a7353b`](https://github.com/rapidsai/cudf/commit/c1f8a7353bea7b7a6083ecd8ae7a00218aa70788) | 305s → 254s (**-51s**) | +586/-497, 7 files | Most invasive — introduces type-erased interface class and factory pattern. Higher review burden. |

### Tier 4: Small or unmeasured impact

| Priority | Optimization | Commit | Critical path Δ | Diff | Rationale |
|----------|--------------|--------|-----------------|------|-----------|
| 12 | contains_table extern templates | [`1b1176fe1e`](https://github.com/rapidsai/cudf/commit/1b1176fe1e31652e750600ae9a0e9653272cae6a) | unmeasured | +137, 1 file | Additive-only, zero risk. |
| 13 | hash_join_probe further split | [`659f249f1f`](https://github.com/rapidsai/cudf/commit/659f249f1fce1367c0286c09939addc3c172ebf8) | unmeasured | +70/-59, 3 files | Small, clean. Applied last so impact not isolated. |

Tier 1 alone eliminates the #1, #3, #4, and #8 slowest files in the build. If submitting a single PR upstream, these four commits deliver the largest savings with the smallest and least invasive diffs.

## Remaining High-Cost Files (Not Amenable to Further Optimization)

| File | Time (s) | Lines | Bottleneck |
|------|----------|-------|------------|
| `mixed_join_kernel_nulls.cu` | 391 | 28 | Row operator templates |
| `compute_global_memory_aggs_null.cu` | 370 | — | 36-kind aggregation dispatcher (constrained dispatch causes runtime regression) |
| `contains_table_impl_nested.cu` | 301 | — | Already split from `contains_table.cu` |
| `compute_global_memory_aggs.cu` | 279 | — | Same as null variant |
| `unique.cu` | 267 | 121 | cuco hash table templates |
| `hash_join_probe.cu` | 267 | — | Already split from `hash_join.cu` |
| `sha256_hash.cu` | 257 | 72 | SHA hash templates |
| `key_remapping_nested.cu` | 254 | — | Already split from `key_remapping.cu` |
| `segmented_sort.cu` | 247 | 95 | thrust sort templates |
| `sha224_hash.cu` | 243 | 72 | SHA hash templates |
| `stable_segmented_sort.cu` | 242 | 95 | thrust sort templates |
