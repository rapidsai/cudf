/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Type aliases for the cuco hash table ref types and equality comparators
// used across hash join probe kernels.  There are 3 dispatch paths:
// primitive, nested, non-nested.

#pragma once

#include "dispatch.cuh"
#include "hash_join_impl.cuh"

#include <cuco/static_multiset.cuh>

namespace cudf::detail {

// --- Equality types from the 3 dispatch paths ---

using primitive_equality_t = primitive_pair_equal;

using nested_equality_t = pair_equal<row::equality::strong_index_comparator_adapter<
  row::equality::device_row_comparator<true, cudf::nullate::DYNAMIC>>>;

using flat_equality_t = pair_equal<row::equality::strong_index_comparator_adapter<
  row::equality::device_row_comparator<false, cudf::nullate::DYNAMIC>>>;

// --- Count ref types (used by count_each kernel) ---

template <typename Equality>
using count_ref_t =
  decltype(std::declval<hash_table_t const&>()
             .ref(cuco::op::count)
             .rebind_key_eq(std::declval<Equality>())
             .rebind_hash_function(std::declval<hash_table_t const&>().hash_function()));

using primitive_count_ref_t = count_ref_t<primitive_equality_t>;
using nested_count_ref_t    = count_ref_t<nested_equality_t>;
using flat_count_ref_t      = count_ref_t<flat_equality_t>;

}  // namespace cudf::detail
