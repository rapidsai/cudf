---
name: review-cudf
description: Review GitHub pull requests for the cudf project. Use when the user invokes /review-cudf with a GitHub PR link, or asks to review a PR, code changes, or diff for cudf.
---

# Review cuDF Pull Request

## Instructions

When the user provides a PR link (e.g. `/review-cudf https://github.com/rapidsai/cudf/pull/12345`):

1. **Fetch PR metadata and diff**

```bash
gh pr view <PR_NUMBER> --repo rapidsai/cudf --json title,body,files,additions,deletions,baseRefName,headRefName
gh pr diff <PR_NUMBER> --repo rapidsai/cudf
```

Hint: Ensure `GH_TOKEN` (or GitHub CLI auth) is already configured in the environment (for example via your secret manager) so `gh` can authenticate and bypass rate limits; do not run `gh auth token` from within the agent. If no token is available, use alternative methods.

2. **Fetch review comments already posted**

```bash
gh api repos/rapidsai/cudf/pulls/<PR_NUMBER>/comments
gh api repos/rapidsai/cudf/pulls/<PR_NUMBER>/reviews
```

3. **Read the Developer Guide** (`cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md`) — it is the authoritative reference for libcudf conventions. All rules in the guide apply during review. The checklist below calls out the most review-relevant rules and adds items **not** covered by the guide.

4. **Analyze the changes** against the checklist below, reading relevant source files as needed for context.

5. **Produce a structured review** using the output format at the bottom.

6. **Dump the structured review** to `.cursor/reviews/<PR NUMBER>/review.md`

---

## Review Checklist

> Items marked **(guide)** are covered in detail in the Developer Guide — verify compliance against the guide. Items without the tag are review-specific or derived from codebase patterns.

### Correctness & Logic

- Core logic implemented by algorithms is clear, correct and coherent.
- Optimal algorithms and data structures have been employed. For example, use `std::unordered_set` instead of a `std::vector` for mass-queries. 
- GPU-accelerated algorithms via Thrust, CUB and cuCollections are used for compute-intensive workloads.
- `cudf::detail::host_worker_pool()` is used for compute or memory intensive parallelizable CPU algorithms.
- No unnecessary memory usage or leaks, no excessive defensive programming.
- Algorithms handle edge cases (empty input, single-row, nulls, sliced columns with nonzero offset, unintended fallthrough logic).
- For recursive and branched algorithms, trace the recursion/branch to ensure no pitfalls, unintended fallthroughs, or unhandled cases.
- Google Tests and/or Python tests must cover all possible edge cases.
- No off-by-one errors in index arithmetic or kernel launch bounds.
- Proper handling of nullable columns and validity masks.
- Correct use of `cudf::size_type` (signed 32-bit) for sizes, offsets, indices. **(guide: "cudf::size_type")**
- Stream ordering preserved across the API flow. **(guide: "Streams")**
- Null values of fixed-width columns are undefined — code must not assume null rows contain any particular value; use the validity mask. **(guide: "Null values of fixed-width columns are undefined")**
- Use `cudf::have_same_types()` for data type comparison, not `a.type() == b.type()`. **(guide: "Comparing Data Types")**
- Nested type columns (LIST, STRUCT) must be sanitized per guide rules. **(guide: "libcudf expects nested types to have sanitized null masks")**
- Do not access the offsets child of an empty strings or lists column. **(guide: "Empty Columns")**

### Naming & Code Duplication
- No significant code duplication; reusable logic must be refactored into common helper functions.
- Function and variable names are meaningful — not too vague or verbose.
- No single-letter variable names except loop indices (`i`, `j`, `k`) or thread IDs (`t`, `tid`).
- Private member variables prefixed with underscore. **(guide: "Code and Documentation Style")**

### API & Design (libcudf C++)

Verify compliance with the guide sections: **"Directory Structure and File Naming"**, **"Streams"**, **"Default Parameters"**, **"NVTX Ranges"**, **"Input/Output Style"**, **"Multiple Return Values"**, **"Namespaces"**, **"Error Handling"**, **"Spans"**, and **"Deprecating and Removing Code"**. Key points to watch for:

- Public APIs in `cpp/include/cudf/` with `CUDF_EXPORT`; detail headers in `include/cudf/detail/` or `include/cudf/<sub>/detail/`.
- Stream and MR as last two parameters (stream before MR); public defaults, no defaults in detail APIs.
- Views as input, `unique_ptr` as output; `std::pair` for multiple returns (not `std::tuple`).
- `CUDF_EXPECTS` / `CUDF_FAIL` / `CUDF_UNREACHABLE` used correctly; `CUDF_EXPECTS` condition must be a pure predicate.
- Public functions: `CUDF_FUNC_RANGE()` then delegate to `detail::`.
- `cudaDeviceSynchronize()` never used; sync only via `stream.synchronize()` when required.

Additional review-specific items not in the guide:
- No raw owning pointers; use `std::unique_ptr`, `std::shared_ptr`, `std::reference_wrapper`.
- Prefer pinned memory/vectors for small H2D and H2D transfers via `cudf::make_pinned_vector` instead of `cudf::make_host_vector`.
- Prefer `span` versions of constructors for `cudf::make_pinned_vector` and `cudf::make_host_vector`.
- Functions defined in headers (e.g. templates) must be `inline`.
- Anonymous namespaces for single-TU helpers; never in headers.
- Prefer `host_span`/`device_span` over owning vectors unless transferring ownership via rvalue move.
- Prefer modern C++ primitives allowed by the current libcudf standard (see `cpp/CMakeLists.txt`): concepts, `requires`, `std::ranges`, `std::for_each`, `std::transform` over manual implementations and raw loops. Range-for like `for (auto elem : vec)` is fine.
- Use `static_assert` with a clear message to prevent accidental template misuse.

### Memory Allocation & Management

Verify compliance with guide sections: **"Memory Allocation"** (Output Memory, Temporary Memory, Memory Management) and **"Memory Copies"**. Key points:

- Returned memory uses the passed-in MR; temporary memory uses `cudf::get_current_device_resource_ref()`.
- Prefer `rmm::device_uvector` over `device_vector`/`thrust::device_vector`.
- Use `cudf::detail::device_scalar<T>` (not `rmm::device_scalar<T>`).
- Prefer `cudf::detail::cuda_memcpy_async` / `memcpy_async` / `memcpy_batch_async` over raw `cudaMemcpyAsync`.
- Async memcpy staging buffers must outlive the copy — use `make_pinned_vector_async`; pageable buffers need `stream.synchronize()` first.

### Style & Formatting

Verify compliance with guide sections: **"Code and Documentation Style and Formatting"**, **"C++ Guidelines"**, and **"File extensions"**. Additional items from codebase patterns:

- CUDA kernels use `CUDF_KERNEL` macro (not raw `__global__`), preferably with `__launch_bounds__`.
- Use `cuda::std::` types/algorithms (`cuda::std::min`, `cuda::std::distance`, `cuda::std::pair`) instead of `std::` in device code.
- Use `cuda::make_constant_iterator` over `thrust::make_constant_iterator` for device-side constant iterators.
- Prefer `cuda::proclaim_return_type<T>(lambda)` when passing device lambdas to `make_counting_transform_iterator`.

### Includes

Verify compliance with guide section: **"Includes"**. Watch especially for:
- Unnecessary includes in headers.
- Wrong bracket style (`<>` vs `""`).
- Internal `src/` headers included from tests or public headers.

### Documentation (Doxygen)
- Public API functions and classes have `/** ... */` doxygen comments.
- Use `@brief`, `@param`, `@return`, `@throw`, `@tparam` tags. Document all of these when a function, struct, or functor is added or modified unless it's trivial and in the `detail` namespace (just `@brief` is sufficient there).
- Copyright header: `SPDX-FileCopyrightText: Copyright (c) <year>, NVIDIA CORPORATION.` with `SPDX-License-Identifier: Apache-2.0`.
- Copyright year span updated if the file was modified.

### Deprecations & Removals

Verify compliance with guide section: **"Deprecating and Removing Code"**. Ensure `[[deprecated]]`, `@deprecated`, and correct PR labels ("deprecation" / "breaking").

### Testing

Refer to the **Testing Guide** (`cpp/doxygen/developer_guide/TESTING.md`) for full conventions. Key review checks:

- Tests use `#include <cudf_test/cudf_gtest.hpp>` (never raw `gtest/gtest.h`).
- Test code is in the **global namespace** (no `using namespace cudf;`).
- Tests cover: empty input, null values, sliced columns, boundary sizes (around warp size 32), multi-block sizes.
- Strings tests include non-ASCII UTF-8 characters.
- Decimal types are in `FixedWidthTypes` but not `NumericTypes`; verify correct type list usage.
- Prefer `rmm::device_uvector` and `column_wrapper` over `thrust::device_vector`.
- Use `ASSERT_*` or `EXPECT_*` correctly for exception safety.

### Benchmarks

Refer to the **Benchmarking Guide** (`cpp/doxygen/developer_guide/BENCHMARKING.md`). Key review checks:

- New benchmarks use NVBench (not Google Benchmark).
- Benchmark source mirrors the feature path (`cpp/benchmarks/<feature>/`).
- Prefer `.cpp` over `.cu` for benchmark files when possible.
- NVTX ranges inserted for all public and work-heavy internal functions via `CUDF_FUNC_RANGE()` or `cudf::scoped_range`.

### Performance
- Prefer STL/Thrust/CUB algorithms over raw loops and raw kernels. **(guide: "C++ Guidelines")**
- Use `rmm::exec_policy_nosync(stream)` for all Thrust device execution. **(guide: "Thrust Execution Policy")**
- Avoid unnecessary host-device synchronization.
- Avoid unnecessary copies; prefer views.
- Avoid multiple levels of `type_dispatcher`. **(guide: "Avoid Multiple Type Dispatch")**

### CUDA-Specific
- Kernels check bounds correctly.
- Shared memory usage doesn't exceed device limits.
- No race conditions in concurrent kernel access patterns.
- Stream and MR parameters propagated across all internal APIs for stream-ordered memory management and kernel launches.

### Type Dispatch Patterns
- Dispatch functors use C++20 `requires` clauses (preferred) or `CUDF_ENABLE_IF` for type-gating `operator()` overloads.
- Unsupported type overloads call `CUDF_FAIL` or `CUDF_UNREACHABLE` as appropriate.
- Functors may include a `static constexpr bool is_supported()` helper for compile-time type filtering.

### Column Construction Patterns
- Use `make_empty_column(type)` for early returns when `size == 0`.
- Use `make_fixed_width_column` / `make_numeric_column` with the appropriate `mask_state` (`UNALLOCATED`, `ALL_NULL`, `ALL_VALID`).
- Use `cudf::detail::copy_bitmask(col, stream, mr)` when output nullability mirrors input.
- Use `cudf::detail::bitmask_and(table_view({col1, col2}), stream, mr)` to combine null masks; returns `{null_mask, null_count}`.
- Pass `rmm::device_buffer{stream}` (not a null pointer) when constructing a column with `null_count == 0`.
- Strings columns use the two-phase approach via `make_strings_children` (computes sizes then fills chars). The functor stores `size_type* d_sizes`, `char* d_chars`, and `input_offsetalator d_offsets`; when `d_chars == nullptr` it computes sizes, otherwise it writes characters.
- Use `cudf::strings::detail::make_offsets_child_column` (returns `{offsets_col, total_bytes}`) for strings; use `cudf::detail::make_offsets_child_column` for non-strings lists.
- Use `cudf::detail::offsetalator_factory::make_input_iterator` for type-erased offset access supporting both INT32 and INT64 offsets.
- Use `cudf::detail::make_counting_transform_iterator` instead of `thrust::make_transform_iterator(thrust::counting_iterator(0), fn)`.
- `column_device_view::create(col_view, stream)` returns a smart pointer; dereference with `*d_col` when passing to kernels or Thrust.
- For strings columns, pass `.parent()` to `column_device_view::create`.

### Python Bindings (if applicable)
- Cython bindings match the C++ API.
- Proper GIL handling.
- Docstrings present and accurate.
- Python code conforms to cudf Python and general Pythonic standards.

### Python Testing (if applicable)
- New pytests match standards and patterns across other tests in the file.
- New pytest files conform to standards and patterns of existing Python test files in that directory.

### Nits & Comments
- No excessive or trivial comments.
- No narration of obvious logic.
- Meaningful comments around non-obvious, complex, or unintuitive logic.
- Meaningful comments around function calls that change state or do multiple things.

---

## Reference Material

The Developer Guide is the primary reference. Read it first for any section marked **(guide)** above.

| Usage  | Topic | Path |
|--------|-------|------|
| HIGH   | Developer guide | `cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md` |
| MEDIUM | Testing guide | `cpp/doxygen/developer_guide/TESTING.md` |
| LOW | Benchmarking guide | `cpp/doxygen/developer_guide/BENCHMARKING.md` |
| LOW    | Documentation guide | `cpp/doxygen/developer_guide/DOCUMENTATION.md` |
| LOW | Profiling guide | `cpp/doxygen/developer_guide/PROFILING.md` |

Online libcudf API docs if needed: https://docs.rapids.ai/api/cudf/nightly/libcudf_docs/

---

## Output Format

Structure your review as follows:

```markdown
## PR Review: <PR title>

**PR:** <link>
**Summary:** <1-2 sentence summary of what the PR does>

### Findings

#### Critical
- **[file:line]** Description of issue that must be fixed before merge.

#### Suggestions
- **[file:line]** Description of improvement to consider.

#### Nits
- **[file:line]** Minor style or formatting issue. Keep these minimal, don't suggest adding comments around every line of code or obvious logic.

#### Highlights
- Highlight well-written code, good test coverage, or clever solutions.

### Verdict
One of: **Approve**, **Request Changes**, or **Comment**
With a brief justification.
```

If there are no findings in a category, omit that category.
