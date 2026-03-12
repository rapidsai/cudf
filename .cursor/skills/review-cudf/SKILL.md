---
name: review-cudf
description: Review GitHub pull requests for the cudf project. Use when the user invokes /review-cudf with a GitHub PR link, or asks to review a PR, code changes, or diff for cudf.
---

# Review cuDF Pull Request

1. **Fetch PR metadata and diff**

```bash
gh pr view <PR_NUMBER> --repo rapidsai/cudf --json title,body,files,additions,deletions,baseRefName,headRefName
gh pr diff <PR_NUMBER> --repo rapidsai/cudf
```

Hint: Ensure `GH_TOKEN` (or GitHub CLI auth) is already configured in the environment (for example via your secret manager) so `gh` can authenticate and bypass rate limits; do not run `gh auth token` from within the agent. If no token is available, use alternative methods.

2. **Fetch review comments already posted** for context on what's already been suggested and need not be repeated.

3. **Read the Developer Guide** (`cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md`) — it is the authoritative reference for libcudf conventions. All rules in the guide apply during review. The checklist below calls out the most review-relevant rules and adds items **not** covered by the guide.

4. **Analyze the changes** against the checklist below, reading relevant source files as needed for context.

5. **Produce a structured review** using the output format at the bottom.

6. **Dump the structured review** to `.cursor/reviews/<PR NUMBER>/review.md`

---

## Review Checklist

> Items marked **(guide: Section Name(s))** are covered in detail in the Developer Guide — verify compliance against the mentioned section(s) of the Developer Guide. Items without the tag are review-specific or derived from codebase patterns.

### Correctness & Logic
- Core logic implemented by algorithms is correct and coherent.
- No unnecessary memory usage or leaks.
- Recursive and branched algorithms have no pitfalls, unintended fallthroughs, or unhandled cases. Trace such algorithms.
- Algorithms cover all possible edge cases such as empty input and nulls, without being excessively defensive.
- Offsets child of an empty string or list column is not accessed. **(guide: "Empty Columns")**
- Proper null handling for fixed-width columns. **(guide: ## Null values of fixed-width columns are undefined)**
- Nested type columns (LIST, STRUCT) are sanitized. **(guide: "libcudf expects nested types to have sanitized null masks")**
- Google Tests and/or Python tests cover all possible edge cases; no duplicated tests.
- No off-by-one errors in index arithmetic or kernel launch bounds.
- Correct use of `cudf::size_type` (signed 32-bit) for sizes, offsets, indices. **(guide: "cudf::size_type")**
- Stream ordering preserved across the API flow. **(guide: "Streams")**
- Use `cudf::have_same_types()` for data type comparison, not `a.type() == b.type()`. **(guide: "Comparing Data Types")**

### Performance Optimization
- Optimal algorithms and data structures have been employed. For example, use `unordered_set`s instead of `vector`s for frequent lookups.
- Ensure compute-intensive algorithms are GPU-accelerated when possible.
- Ensure Host-side compute or memory intensive algorithms are accelerated using `cudf::detail::host_worker_pool()` when possible.
- Use CUB (most preferred)/Thrust/STL algorithms over raw loops and raw kernels. **(guide: "C++ Guidelines")**
- Use `rmm::exec_policy_nosync(stream)` for all Thrust device execution. **(guide: "Thrust Execution Policy")**
- No unnecessary host-device synchronization.
- Avoid multiple levels of `type_dispatcher`. **(guide: "Avoid Multiple Type Dispatch")**

### CUDA-Specific
- CUDA kernels check bounds correctly.
- Shared memory usage doesn't exceed device limits.
- No race conditions and out of bound memory accesses in concurrent kernel access patterns.
- Stream and MR parameters are propagated across all internal APIs for stream-ordered memory management and kernel launches.
- Modern CUDA C++ primitives and patterns using Thrust/CUB (NVIDIA CCCL) and cooperative groups preferred over C-like versions. For example, `cuda::std::popcount` over `__popc`, `cg::thread_block::thread_rank()` over `threadIdx.x.

### Type Dispatch Patterns
- New dispatch functors prefer C++20 `requires` clauses over `CUDF_ENABLE_IF` for type-gating `operator()` overloads.
- Unsupported type overloads call `CUDF_FAIL` or `CUDF_UNREACHABLE` as appropriate.
- Functors may include a `static constexpr bool is_supported()` helper for compile-time type filtering.

### Column Construction Patterns
- Use `make_empty_column` to construct empty columns for early returns.
- Use `cudf::make_column_from_scalar` to construct a column filled with the same value.
- Use `cudf::detail::copy_bitmask` when output nullability mirrors the input.
- Use `cudf::detail::bitmask_and` to combine null masks.
- Use `rmm::device_buffer{0, stream}` (not a null pointer) when constructing a non-nullable column.
- Strings columns may use a two-phase approach via `make_strings_children`.
- Use `cudf::strings::detail::make_offsets_child_column` for strings; use `cudf::detail::make_offsets_child_column` for non-strings lists.
- Use `cudf::detail::offsetalator_factory::make_input_iterator` for type-erased offset access supporting both INT32 and INT64 offsets.
- Use `cudf::detail::make_counting_transform_iterator` instead of `thrust::make_transform_iterator(thrust::counting_iterator(0), fn)`.
- For strings columns, pass `.parent()` to `column_device_view::create`.

### Naming & Code Duplication
- No significant code duplication; reusable logic must be refactored into common helper functions.
- Function and variable names are meaningful — not too vague or verbose.
- No single-letter variable names except loop indices (`i`, `j`, `k`) or thread IDs (`t`, `tid`).
- Private member variables prefixed with underscore. **(guide: "Code and Documentation Style")**

### API & Design (libcudf C++)

Verify compliance with the developer guide sections: **(guide: "Directory Structure and File Naming"**, **"Streams"**, **"Default Parameters"**, **"NVTX Ranges"**, **"Input/Output Style"**, **"Multiple Return Values"**, **"Namespaces"**, **"Error Handling"**, **"Spans"**, and **"Deprecating and Removing Code")**. Key review checks:

- Public APIs in `cpp/include/cudf/` with `CUDF_EXPORT`; detail headers in `include/cudf/detail/` or `include/cudf/<sub>/detail/`.
- Stream and MR as last two parameters (stream before MR); public defaults, no defaults in detail APIs.
- `CUDF_EXPECTS` / `CUDF_FAIL` / `CUDF_UNREACHABLE` are used correctly; `CUDF_EXPECTS` condition must be a pure predicate.
- Public functions: `CUDF_FUNC_RANGE()` then delegate to `detail::`.
- `cudaDeviceSynchronize()` is never ever used; sync only via `stream.synchronize()` when required.

Additional key checks not in the guide:
- No raw owning pointers; use `std::unique_ptr`, `std::shared_ptr`, `std::reference_wrapper`.
- Prefer pinned memory/vectors for small H2D and H2D transfers via `cudf::make_pinned_vector` instead of `cudf::make_host_vector`.
- Prefer `span` versions of constructors for `cudf::make_pinned_vector` and `cudf::make_host_vector`.
- Functions defined in headers (e.g. templates) must be `inline`.
- Anonymous namespaces for single-TU helpers; never in headers.
- Use `host_span`/`device_span` ; no owning vectors passed around by copy/reference unless explicitly moved (transferring ownership).
- Use modern C++20 primitives such as `concepts`, `std::ranges`, `std::transform` over manual implementations and raw loops; Range-for loops are fine.
- Use `static_assert` with a clear message to prevent accidental template misuse.

### Memory Allocation & Management

Verify compliance with the developer guide sections: **(guide: "Memory Allocation)"** (subsections: Output Memory, Temporary Memory, Memory Management) and **(guide:"Memory Copies")**. Key checks:

- Returned memory uses the passed-in MR; temporary memory uses `cudf::get_current_device_resource_ref()`.
- Use `rmm::device_uvector` or `column_wrapper` (no `device_vector`/`thrust::device_vector`).
- Use `cudf::detail::device_scalar<T>` (not `rmm::device_scalar<T>`).
- Use `cudf::detail::cuda_memcpy_async` / `memcpy_async` / `memcpy_batch_async` (no raw `cudaMemcpyAsync`).
- Async memcpy staging buffers must outlive the copy — use `make_pinned_vector_async`; pageable buffers need `stream.synchronize()` first.

### Style & Formatting

Verify compliance with the developer guide sections: **(guide: "Code and Documentation Style and Formatting"**, **"C++ Guidelines"**, and **"File extensions")**. Additional key checks from codebase patterns:

- CUDA kernels use `CUDF_KERNEL` macro (not raw `__global__`), preferably with `__launch_bounds__`.
- Use `cuda::std::` types/algorithms (`cuda::std::min`, `cuda::std::distance`, `cuda::std::pair`) instead of `std::` in device code.
- Use `cuda::make_constant_iterator` over `thrust::make_constant_iterator` for device-side constant iterators.
- Use `cuda::proclaim_return_type<T>(lambda)` when passing device lambdas to `make_counting_transform_iterator`.

### Includes

Verify compliance with the developer guide section: **(guide: "Includes")**. Key review checks:
- No unnecessary includes in headers.
- Bracket style (`<>` vs `""`).

### Documentation & Doxygen
Verify compliance with the documentation guide. Key checks:
- Public API functions and classes have `/** ... */` doxygen comments.
- `@brief`, `@param`, `@return`, `@throw`, `@tparam` tags used and documented unless it's trivial and in the `detail` namespace (just `@brief` is sufficient here).

### Deprecations & Removals

Verify compliance with the developer guide section: **(guide:"Deprecating and Removing Code")**. Ensure `[[deprecated]]`, `@deprecated`, and correct PR labels ("deprecation" / "breaking").

### Testing

Refer to the **Testing Guide** (`cpp/doxygen/developer_guide/TESTING.md`) for full conventions. Key review checks:

- Tests use `#include <cudf_test/cudf_gtest.hpp>` (never raw `gtest/gtest.h`).
- Test code is in the **global namespace** (no `using namespace cudf;`).
- Tests cover: empty input, null values, sliced columns, boundary sizes, multi-block sizes when applicable.
- Strings tests include non-ASCII UTF-8 characters.
- Decimal types are in `FixedWidthTypes` but not `NumericTypes`; verify correct type list usage.
- `ASSERT_*` or `EXPECT_*` correctly used for exception safety.

### Benchmarks

Refer to the **Benchmarking Guide** (`cpp/doxygen/developer_guide/BENCHMARKING.md`). Key review checks:

- New benchmarks use NVBench (not Google Benchmark).
- Benchmark source mirrors the feature path (`cpp/benchmarks/<feature>/`).
- Prefer `.cpp` over `.cu` for benchmark files when possible.

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

| Topic | Path |
|-------|------|
| Developer guide | `cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md` |
| Testing guide | `cpp/doxygen/developer_guide/TESTING.md` |
| Benchmarking guide | `cpp/doxygen/developer_guide/BENCHMARKING.md` |
| Documentation guide | `cpp/doxygen/developer_guide/DOCUMENTATION.md` |
| Profiling guide | `cpp/doxygen/developer_guide/PROFILING.md` |

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
