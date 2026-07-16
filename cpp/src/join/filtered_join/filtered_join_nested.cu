/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "filtered_join_common.cuh"

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuco/operator.hpp>
#include <cuco/static_set_ref.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

namespace cudf::detail {

void filtered_join::insert_right_table_nested(rmm::cuda_stream_view stream)
{
  auto const comparator =
    cudf::detail::row::equality::self_comparator{_preprocessed_right}.equal_to<true>(
      nullate::YES{},
      _nulls_equal,
      cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
  cuco::static_set_ref set_ref{empty_sentinel_key,
                               insertion_adapter{comparator},
                               nested_probing_scheme{},
                               cuco::thread_scope_device,
                               _bucket_storage.ref()};
  auto hasher =
    cudf::detail::row::hash::row_hasher{_preprocessed_right}.device_hasher(nullate::YES{});
  auto const iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, key_pair_fn<lhs_index_type, decltype(hasher)>{hasher});
  insert_right_table<nested_probing_scheme::cg_size>(
    iter, set_ref.rebind_operators(cuco::insert), stream);
}

}  // namespace cudf::detail
