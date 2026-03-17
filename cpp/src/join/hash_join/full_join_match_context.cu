/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "match_context_common.cuh"

#include <cudf/detail/nvtx/ranges.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/transform_output_iterator.h>

namespace cudf::detail {

template <typename Hasher>
cudf::join_match_context hash_join<Hasher>::full_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"hash_join::full_join_match_context"};

  auto match_counts =
    std::make_unique<rmm::device_uvector<size_type>>(probe.num_rows(), stream, mr);

  if (_is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream), match_counts->begin(), match_counts->end(), 1);
  } else {
    auto transform = [] __device__(size_type count) { return count == 0 ? 1 : count; };
    auto transformed_output =
      thrust::make_transform_output_iterator(match_counts->begin(), transform);
    cudf::detail::compute_match_counts(_build,
                                       _preprocessed_build,
                                       _hash_table,
                                       _has_nulls,
                                       _nulls_equal,
                                       probe,
                                       transformed_output,
                                       stream);
  }

  return cudf::join_match_context{probe, std::move(match_counts)};
}

template cudf::join_match_context hash_join<hash_join_hasher>::full_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const;

}  // namespace cudf::detail
