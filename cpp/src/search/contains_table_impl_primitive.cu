/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "contains_table_impl.cuh"

#include <cudf/detail/row_operator/primitive_row_operators.cuh>

namespace cudf::detail {

// Explicit instantiation for perform_contains with primitive row operations
using primitive_hasher_adapter_type = hasher_adapter<cudf::detail::row::primitive::row_hasher<>,
                                                     cudf::detail::row::primitive::row_hasher<>>;

using primitive_comparator_adapter_type =
  comparator_adapter<cudf::detail::row::primitive::row_equality_comparator,
                     cudf::detail::row::primitive::row_equality_comparator>;

template void perform_contains(
  table_view const& haystack,
  table_view const& needles,
  bool haystack_has_nulls,
  bool needles_has_nulls,
  null_equality compare_nulls,
  primitive_comparator_adapter_type const& d_equal,
  cuco::linear_probing<1, primitive_hasher_adapter_type> const& probing_scheme,
  rmm::device_uvector<bool>& contained,
  rmm::cuda_stream_view stream);

}  // namespace cudf::detail
