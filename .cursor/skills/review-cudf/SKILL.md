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

3. **Analyze the changes** against the checklist below, reading relevant source files as needed for context.

4. **Produce a structured review** using the output format at the bottom.

5. **Dump the structured review** to `/home/coder/cudf/.cursor/reviews/<PR NUMBER>/review.md`

---

## Review Checklist

### Correctness & Logic
- Algorithms handle edge cases (empty input, single-row, nulls, sliced columns with nonzero offset, unintended fallthrough logic).
- For recursive and branched algorithms, trace the recursion/branch to ensure no pitfalls, unintended fallthroughs, or unhandled cases.
- Google Tests and/or Python tests must cover all possible edge cases.
- No off-by-one errors in index arithmetic or kernel launch bounds.
- Proper handling of nullable columns and validity masks.
- Correct use of `cudf::size_type` (signed 32-bit) for sizes, offsets, indices.
- Stream orderness preserved across the API flow.
- Null values of fixed-width columns are undefined — code must not assume null rows contain any particular value (e.g., zero); use the validity mask to determine nullness.
- When comparing data types of two columns/scalars, use `cudf::have_same_types()` rather than `a.type() == b.type()` — direct comparison mishandles nested types.
- Nested type columns (LIST, STRUCT) must be sanitized: null list elements should have equal start/end offsets; null struct rows must have null fields; nulls should only be at the parent level for compound columns.
- Do not access the offsets child of an empty strings or lists column — this is undefined behavior.

### Naming & Code Duplication
- No significant code duplication; reusable logic must be refactored into common helper functions.
- Function and variable names are meaningful — not too vague or verbose.
- No single-letter variable names except loop indices (`i`, `j`, `k`) or thread IDs (`t`, `tid`).
- Private member variables are prefixed with an underscore (`_rating`, `_column`).

### API & Design (libcudf C++)
- Public APIs live in `cpp/include/cudf/` with `CUDF_EXPORT` on the `cudf` namespace. The `cudf` namespace must not be nested when `CUDF_EXPORT` is applied.
- Internal detail headers go in `include/cudf/detail/` or `include/cudf/<sub-namespace>/detail/`.
- Functions take views as input (`column_view`, `table_view`) and return `std::unique_ptr<column>` or `std::unique_ptr<table>`.
- Stream and MR parameters are the last two parameters, in that order. Stream comes just before MR.
- Public APIs default stream to `cudf::get_default_stream()` and MR to `cudf::get_current_device_resource_ref()`.
- Detail APIs must not have default parameters for stream or MR.
- No raw owning pointers; use `std::unique_ptr`, `std::shared_ptr`, `std::reference_wrapper`.
- Proper exception handling via `CUDF_EXPECTS`, `CUDF_FAIL`, `CUDF_UNREACHABLE`. Prefer specific exception types (`std::invalid_argument`, `std::out_of_range`) as the third arg to `CUDF_EXPECTS` when appropriate.
- Prefer `CUDF_EXPECTS(condition, msg)` over `if (!condition) { CUDF_FAIL(msg); }` when the condition has no side effects. Otherwise, evaluate the `condition` in a variable separately and use that in `CUDF_EXPECTS` 
- The condition (first argument) of `CUDF_EXPECTS` must be a pure predicate — capture side-effecting results in a variable first.
- `CUDF_UNREACHABLE` (not `CUDF_FAIL`) for template code paths that are statically unreachable but cannot be removed due to template instantiation.
- Input validation is limited to column/table sizes and data types; libcudf does not validate data contents (integer overflow, 2GB limit, etc.).
- Use RAII to scope temporary pinned/device memory and mutexes.
- Multiple return values use `std::pair` (not `std::tuple` — Cython does not support `std::tuple`). Multiple values of the same type use `std::vector<T>`.
- Internal functions used only in a single translation unit go in an anonymous namespace in the `.cpp`/`.cu` file. Anonymous namespaces must never appear in header files.
- Functions defined in header such as templated ones must be `inline`d.
- Functions reused across translation units go in the `detail` namespace with a header in `include/cudf/detail/`.
- Public API functions call `CUDF_FUNC_RANGE()` then delegate entirely to a `detail::` function with the same signature (minus defaults).
- `cudaDeviceSynchronize()` must never be used. Synchronize only when necessary (e.g., before returning a non-pointer host value from an async copy), using `stream.synchronize()`.
- Forking and joining the input stream to launch multiple kernels across the same or multiple threads is fine.
- Prefer accepting `host_span`/`device_span` over owning vectors unless when transferring ownership by explicitly moving via rvalue.
- Spans are lightweight views — pass by value, apply `const` to the element type (`span<T const>`), not to the span itself.
- Prefer modern C++ primitives allowed by the current libcudf CXX standard (can be found in `cpp/CMakeLists.txt`). For example, use C++ concepts, `requires()`, `std::ranges`, `std::for_each`, `std::transform` instead of manual implementations and raw for loops. For loops like `for (auto elem : vector)` are fine.
- Use `static_assert` with a clear message when possible to avoid accidental call of a templated function with a wrong tparam or type.
- Pass in views and spans instead of copies or references of owning objects when possible except when transferring ownership by moving

### Memory Allocation & Management
- All APIs returning device memory (columns, `rmm::device_buffer`, `rmm::device_uvector`) accept stream and MR and use them for those allocations.
- Temporary/scratch memory uses `cudf::get_current_device_resource_ref()` (not the passed-in MR).
- Prefer `rmm::device_uvector` over `rmm::device_vector` or `thrust::device_vector` for new code. Use utility factories in `device_factories.hpp` for initialization.
- Use `cudf::detail::device_scalar<T>` (not `rmm::device_scalar<T>`) for single-element device scalars — it uses pinned host memory for transfers, avoiding implicit synchronization.
- Prefer `cudf::detail::cuda_memcpy_async` and `cudf::detail::memcpy_async` over direct `cudaMemcpyAsync`. Use `cudf::detail::memcpy_batch_async` when copying multiple buffers.
- When using async memcpy with a temporary host staging buffer, the buffer must remain valid until the stream executes the copy. Use `cudf::detail::make_pinned_vector_async` for staging buffers to avoid an explicit sync; `std::vector` (pageable) requires `stream.synchronize()` before the vector goes out of scope.
- The MR type to use is `rmm::device_async_resource_ref` (stream-ordered). Use the more specific `rmm::device_resource_ref`, `rmm::host_resource_ref`, etc. only when the allocation semantics differ.

### Style & Formatting
- snake_case everywhere except template parameters and test/test-case names (PascalCase).
- "East const": `int const x`, not `const int x`.
- Decimal literals use `'` separators every 3 digits (`1'234'567`); hex every 4 (`0x0123'ABCD`).
- `.hpp` for C++ headers, `.cpp` for C++ source, `.cu` for CUDA source, `.cuh` for CUDA headers.
- Only use `.cu`/`.cuh` when device code (`__device__`, Thrust device exec policy) is present.
- `#pragma once` for all headers; no `#ifndef` guards.
- CUDA kernels use the `CUDF_KERNEL` macro (not raw `__global__`), preferably with `__launch_bounds__`.
- Use `cuda::std::` types and algorithms (e.g., `cuda::std::min`, `cuda::std::distance`, `cuda::std::pair`) instead of `std::` in device code.
- Use `cuda::make_constant_iterator` (from `<cuda/iterator>`) over `thrust::make_constant_iterator` for device-side constant iterators.
- Prefer `cuda::proclaim_return_type<T>(lambda)` when passing device lambdas to `make_counting_transform_iterator` to explicitly declare return type.

### Includes
- Grouped by library (cuDF, RMM, Thrust/CUB, STL), separated by blank lines, sorted within groups (nearest to farthest).
- `<>` for all includes except internal `src/` or `test/` headers (use `""`). `cudf_test` and `nvtext` public headers use `<>`.
- No unnecessary includes, especially in headers. Double-check when removing code.
- No relative `..` paths when avoidable.
- Do not include libcudf `src/` internal headers from tests or public headers.

### Documentation (Doxygen)
- Public API functions and classes have `/** ... */` doxygen comments.
- Use `@brief`, `@param`, `@return`, `@throw`, `@tparam` tags. Document all of these whenever a function, struct, or functor is added or modified unless it's trivial and in the `detail` namespace (just `@brief` is sufficient there).
- Copyright header: `SPDX-FileCopyrightText: Copyright (c) <year>, NVIDIA CORPORATION.` with `SPDX-License-Identifier: Apache-2.0`.
- Copyright year span updated if the file was modified.

### Deprecations & Removals
- Pending removals use the `[[deprecated]]` attribute and `@deprecated` Doxygen tag. Replacement API is mentioned in both.
- PRs introducing deprecations are labeled "deprecation"; PRs breaking/removing APIs are labeled "breaking".

### Testing
- Tests use Google Test via `#include <cudf_test/cudf_gtest.hpp>` (never raw `gtest/gtest.h`).
- Test code is in the **global namespace** (no `using namespace cudf;`).
- Tests cover: empty input, null values, sliced columns, boundary sizes (around warp size 32), multi-block sizes.
- Strings tests include non-ASCII UTF-8 characters.
- Decimal types are in `FixedWidthTypes` but not `NumericTypes`; verify correct type list usage.
- Prefer `rmm::device_uvector` and `column_wrapper` over `thrust::device_vector` (allows `.cpp` test files).
- Use `ASSERT_*` or `EXPECT_*` correctly in GoogleTests for exception safety.

### Benchmarks
- New benchmarks use NVBench (not Google Benchmark).
- Benchmark source mirrors the feature path (`cpp/benchmarks/<feature>/`).
- Prefer `.cpp` over `.cu` for benchmark files when possible.
- NVTX ranges inserted for all public and work-heavy internal functions via `CUDF_FUNC_RANGE()` or `cudf::scoped_range`.

### Performance
- Prefer STL/Thrust/CUB algorithms over raw loops and raw kernels.
- Use `rmm::exec_policy_nosync(stream)` for all Thrust device execution (not `rmm::exec_policy`). If a sync is needed, call `stream.synchronize()` explicitly.
- Avoid unnecessary host-device synchronization.
- Avoid unnecessary copies; prefer views.
- Avoid multiple levels of `type_dispatcher` — each level multiplies compile time and object code size quadratically.

### CUDA-Specific
- Kernels check bounds correctly.
- Shared memory usage doesn't exceed device limits.
- No race conditions in concurrent kernel access patterns.
- Stream and MR parameters propagated across all internal APIs for stream-ordered memory management and kernel launches.

### Type Dispatch Patterns
- Dispatch functors use C++20 `requires` clauses (preferred in new code) or `CUDF_ENABLE_IF` for type-gating `operator()` overloads (less preferred).
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
- Use `cudf::detail::offsetalator_factory::make_input_iterator` for type-erased offset access supporting both INT32 and INT64 offsets (large strings).
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

When you need deeper context on project conventions and the checklist sections above, read these files from the repo:

| Usage  | Topic | Path |
|--------|-------|------|
| HIGH   | Developer guide | `cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md` |
| MEDIUM | Testing guide | `cpp/doxygen/developer_guide/TESTING.md` |
| LOW | Benchmarking guide | `cpp/doxygen/developer_guide/BENCHMARKING.md` |
| LOW    | Documentation guide | `cpp/doxygen/developer_guide/DOCUMENTATION.md` |
| LOW | Profiling guide | `cpp/doxygen/developer_guide/PROFILING.md` |

Online API docs: https://docs.rapids.ai/api/cudf/nightly/libcudf_docs/
Unlikely needed: Online NVBench docs: https://github.com/NVIDIA/nvbench/blob/main/docs/benchmarks.md

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
