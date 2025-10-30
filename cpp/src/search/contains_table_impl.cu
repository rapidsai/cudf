/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "contains_table_impl.cuh"

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <cuco/static_set.cuh>

namespace cudf::detail {

using cudf::detail::row::lhs_index_type;
using cudf::detail::row::rhs_index_type;

/**
 * @brief Build a row bitmask for the input table.
 *
 * The output bitmask will have invalid bits corresponding to the input rows having nulls (at
 * any nested level) and vice versa.
 *
 * @param input The input table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A pair of pointer to the output bitmask and the buffer containing the bitmask
 */
std::pair<rmm::device_buffer, bitmask_type const*> build_row_bitmask(table_view const& input,
                                                                     rmm::cuda_stream_view stream)
{
  auto const nullable_columns = get_nullable_columns(input);
  CUDF_EXPECTS(nullable_columns.size() > 0,
               "The input table has nulls thus it should have nullable columns.");

  // If there are more than one nullable column, we compute `bitmask_and` of their null masks.
  // Otherwise, we have only one nullable column and can use its null mask directly.
  if (nullable_columns.size() > 1) {
    auto row_bitmask =
      cudf::detail::bitmask_and(
        table_view{nullable_columns}, stream, cudf::get_current_device_resource_ref())
        .first;
    auto const row_bitmask_ptr = static_cast<bitmask_type const*>(row_bitmask.data());
    return std::pair(std::move(row_bitmask), row_bitmask_ptr);
  }

  return std::pair(rmm::device_buffer{0, stream}, nullable_columns.front().null_mask());
}

// Explicit instantiations for non-nested types (HasNested=false)
using hasher_adapter_t = hasher_adapter<
  cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash, nullate::DYNAMIC>,
  cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                             nullate::DYNAMIC>>;

template void dispatch_nan_comparator<false, hasher_adapter_t>(
  table_view const& haystack,
  table_view const& needles,
  null_equality compare_nulls,
  nan_equality compare_nans,
  bool haystack_has_nulls,
  bool needles_has_nulls,
  bool has_any_nulls,
  cudf::detail::row::equality::self_comparator self_equal,
  cudf::detail::row::equality::two_table_comparator two_table_equal,
  hasher_adapter_t const& d_hasher,
  rmm::device_uvector<bool>& contained,
  rmm::cuda_stream_view stream);

// For HasNested=false (non-nested columns) with nan_equal_comparator
using nan_equal_self_comparator = cudf::detail::row::equality::device_row_comparator<
  false,
  cudf::nullate::DYNAMIC,
  cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

using nan_equal_two_table_comparator =
  cudf::detail::row::equality::strong_index_comparator_adapter<nan_equal_self_comparator>;

using nan_equal_comparator_adapter =
  comparator_adapter<nan_equal_self_comparator, nan_equal_two_table_comparator>;

template void perform_contains(table_view const& haystack,
                               table_view const& needles,
                               bool haystack_has_nulls,
                               bool needles_has_nulls,
                               null_equality compare_nulls,
                               nan_equal_comparator_adapter const& d_equal,
                               cuco::linear_probing<1, hasher_adapter_t> const& probing_scheme,
                               rmm::device_uvector<bool>& contained,
                               rmm::cuda_stream_view stream);

// For HasNested=false (non-nested columns) with nan_unequal_comparator
using nan_unequal_self_comparator = cudf::detail::row::equality::device_row_comparator<
  false,
  cudf::nullate::DYNAMIC,
  cudf::detail::row::equality::physical_equality_comparator>;

using nan_unequal_two_table_comparator =
  cudf::detail::row::equality::strong_index_comparator_adapter<nan_unequal_self_comparator>;

using nan_unequal_comparator_adapter =
  comparator_adapter<nan_unequal_self_comparator, nan_unequal_two_table_comparator>;

template void perform_contains(table_view const& haystack,
                               table_view const& needles,
                               bool haystack_has_nulls,
                               bool needles_has_nulls,
                               null_equality compare_nulls,
                               nan_unequal_comparator_adapter const& d_equal,
                               cuco::linear_probing<1, hasher_adapter_t> const& probing_scheme,
                               rmm::device_uvector<bool>& contained,
                               rmm::cuda_stream_view stream);

}  // namespace cudf::detail
