/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.cuh"
#include "stream_compaction/stream_compaction_common.cuh"

#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/row_operator/equality.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace sort {

sort_groupby_helper::index_vector sort_groupby_helper::group_offsets_nested(
  size_type size, size_type const* sorted_order, rmm::cuda_stream_view stream)
{
  auto group_offsets = index_vector(size + 1, stream);

  auto const comparator  = cudf::detail::row::equality::self_comparator{_keys, stream};
  auto const d_key_equal = comparator.equal_to<true>(
    cudf::nullate::DYNAMIC{cudf::has_nested_nulls(_keys)}, null_equality::EQUAL);
  // Using a temporary buffer for intermediate transform results from the iterator containing
  // the comparator speeds up compile-time significantly without much degradation in
  // runtime performance over using the comparator directly in thrust::unique_copy.
  auto result       = rmm::device_uvector<bool>(size, stream);
  auto const itr    = thrust::make_counting_iterator<size_type>(0);
  auto const row_eq = permuted_row_equality_comparator(d_key_equal, sorted_order);
  auto const ufn    = cudf::detail::unique_copy_fn<decltype(itr), decltype(row_eq)>{
    itr, duplicate_keep_option::KEEP_FIRST, row_eq, size - 1};
  thrust::transform(rmm::exec_policy_nosync(stream), itr, itr + size, result.begin(), ufn);
  auto const result_end = cudf::detail::copy_if(
    itr, itr + size, result.begin(), group_offsets.begin(), cuda::std::identity{}, stream);

  auto const num_groups = cuda::std::distance(group_offsets.begin(), result_end);
  group_offsets.set_element_async(num_groups, size, stream);
  group_offsets.resize(num_groups + 1, stream);

  return group_offsets;
}

}  // namespace sort
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
