/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common.cuh"
#include "dispatch.cuh"
#include "join/join_common_utils.cuh"

#include <cudf/detail/iterator.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/transform_output_iterator.h>

namespace cudf::detail {

struct zero_count_to_one {
  __device__ size_type operator()(size_type count) const { return count == 0 ? 1 : count; }
};

template <typename OutputIterator>
void compute_match_counts(
  table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  cudf::table_view const& probe,
  OutputIterator output_iter,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe, stream);
  auto const probe_table_num_rows = probe.num_rows();

  auto count_matches = [&](auto equality, auto d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});
    hash_table.count_each(iter,
                          iter + probe_table_num_rows,
                          equality,
                          hash_table.hash_function(),
                          output_iter,
                          stream.value());
  };

  dispatch_join_comparator(
    build, probe, preprocessed_build, preprocessed_probe, has_nulls, compare_nulls, count_matches);
}

template <typename Hasher>
template <join_kind Join>
cudf::join_match_context hash_join<Hasher>::join_match_context_impl(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  auto match_counts =
    std::make_unique<rmm::device_uvector<size_type>>(probe.num_rows(), stream, mr);

  if (_is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream),
                 match_counts->begin(),
                 match_counts->end(),
                 Join == join_kind::INNER_JOIN ? 0 : 1);
  } else if constexpr (Join == join_kind::INNER_JOIN) {
    cudf::detail::compute_match_counts(_build,
                                       _preprocessed_build,
                                       _hash_table,
                                       _has_nulls,
                                       _nulls_equal,
                                       probe,
                                       match_counts->begin(),
                                       stream);
  } else {
    auto transformed_output =
      thrust::make_transform_output_iterator(match_counts->begin(), zero_count_to_one{});
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

}  // namespace cudf::detail
