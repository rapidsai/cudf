## Summary

Make `sort_merge_join::inner_join()` and `sort_merge_join::left_join()` thread-safe by eliminating shared mutable state during join operations.

## Problem

The `sort_merge_join` class stored `preprocessed_left` as a member variable that was mutated on every call to `inner_join()` or `left_join()`. This made concurrent calls on the same instance unsafe, as multiple threads would race on writing to the shared state.

## Solution

Follow the same pattern used by `hash_join`: create the preprocessed left table as a **local variable** within each join method rather than storing it as a member. This allows multiple threads to call `inner_join()` and `left_join()` concurrently on the same `sort_merge_join` instance without data races.

## Changes

**Header (`sort_merge_join.hpp`):**
- Added thread safety documentation to the class
- Made `inner_join()` and `left_join()` const methods
- Added `preprocessed_table::create()` static factory method
- Made `preprocessed_right` and `compare_nulls` const members
- Updated `invoke_merge()` and `postprocess_indices()` to take `preprocessed_left` as a parameter

**Implementation (`sort_merge_join.cu`):**
- Implemented `preprocessed_table::create()` factory method
- Refactored constructor to use member initializer list
- `inner_join()` and `left_join()` now create local `preprocessed_left` for thread safety
- `inner_join_match_context()` and `partitioned_inner_join()` continue to use the member (documented as not thread-safe)

**Tests (`join_tests.cpp`):**
- Added `SortMergeJoinThreadSafetyTest` fixture with three tests:
  - `ConcurrentInnerJoins` - 4 threads calling `inner_join()` concurrently
  - `ConcurrentLeftJoins` - 4 threads calling `left_join()` concurrently
  - `StressTestConcurrentJoins` - 8 threads × 10 iterations stress test

## Thread Safety Guarantees

| Method | Thread-Safe |
|--------|-------------|
| `inner_join()` | Yes |
| `left_join()` | Yes |
| `inner_join_match_context()` | No |
| `partitioned_inner_join()` | No |

The partitioned join workflow (`inner_join_match_context` + `partitioned_inner_join`) remains not thread-safe as these methods share state by design. This is documented in the class and method docstrings.
