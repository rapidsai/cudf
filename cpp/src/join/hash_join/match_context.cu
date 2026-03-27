/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "dispatch.cuh"
#include "join/join_common_utils.cuh"

#include <cudf/detail/iterator.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/transform.h>

namespace cudf::detail {

std::unique_ptr<rmm::device_uvector<size_type>> make_join_match_counts(
  table_view const& build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  cudf::detail::hash_table_t const& hash_table,
  bool is_empty,
  bool has_nulls,
  null_equality compare_nulls,
  join_kind join,
  table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto match_counts =
    std::make_unique<rmm::device_uvector<size_type>>(probe.num_rows(), stream, mr);

  if (is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream),
                 match_counts->begin(),
                 match_counts->end(),
                 join == join_kind::INNER_JOIN ? 0 : 1);
    return match_counts;
  }

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
                          match_counts->begin(),
                          stream.value());
  };

  dispatch_join_comparator(
    build, probe, preprocessed_build, preprocessed_probe, has_nulls, compare_nulls, count_matches);

  if (join != join_kind::INNER_JOIN) {
    thrust::transform(rmm::exec_policy_nosync(stream),
                      match_counts->begin(),
                      match_counts->end(),
                      match_counts->begin(),
                      [] __device__(size_type count) { return count == 0 ? 1 : count; });
  }

  return match_counts;
}

}  // namespace cudf::detail
