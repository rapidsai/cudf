/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.cuh"

#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/row_operator/equality.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/unique.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace sort {

sort_groupby_helper::index_vector const& sort_groupby_helper::group_offsets(
  rmm::cuda_stream_view stream)
{
  if (_group_offsets) return *_group_offsets;

  auto const size         = num_keys(stream);
  auto const sorted_order = key_sort_order(stream).data<size_type>();

  if (cudf::detail::has_nested_columns(_keys)) {
    _group_offsets =
      std::make_unique<index_vector>(group_offsets_nested(size, sorted_order, stream));
    return *_group_offsets;
  }

  // Create a temporary variable and only set _group_offsets right before the return.
  // This way, a 2nd (parallel) call to this will not be given a partially created object.
  auto group_offsets = std::make_unique<index_vector>(size + 1, stream);

  auto const comparator  = cudf::detail::row::equality::self_comparator{_keys, stream};
  auto const d_key_equal = comparator.equal_to<false>(
    cudf::nullate::DYNAMIC{cudf::has_nested_nulls(_keys)}, null_equality::EQUAL);
  auto const result_end =
    thrust::unique_copy(rmm::exec_policy_nosync(stream),
                        thrust::counting_iterator<size_type>(0),
                        thrust::counting_iterator<size_type>(size),
                        group_offsets->begin(),
                        permuted_row_equality_comparator(d_key_equal, sorted_order));

  auto const num_groups = cuda::std::distance(group_offsets->begin(), result_end);
  group_offsets->set_element_async(num_groups, size, stream);
  group_offsets->resize(num_groups + 1, stream);

  _group_offsets = std::move(group_offsets);
  return *_group_offsets;
}

}  // namespace sort
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
