/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../join_common_utils.cuh"
#include "retrieve_common.cuh"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/nvtx/ranges.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

#include <memory>

namespace cudf::detail {

std::size_t get_full_join_size(
  cudf::table_view const& build_table,
  cudf::table_view const& probe_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_probe,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::size_t join_size = compute_join_output_size<join_kind::LEFT_JOIN>(build_table,
                                                                         probe_table,
                                                                         preprocessed_build,
                                                                         preprocessed_probe,
                                                                         hash_table,
                                                                         has_nulls,
                                                                         compare_nulls,
                                                                         stream);

  if (join_size == 0) { return join_size; }

  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const out_build_begin =
    thrust::make_transform_output_iterator(right_indices->begin(), output_fn{});

  retrieve_left_join_build_indices(build_table,
                                   probe_table,
                                   preprocessed_build,
                                   preprocessed_probe,
                                   hash_table,
                                   has_nulls,
                                   compare_nulls,
                                   out_build_begin,
                                   stream);

  auto const left_table_row_count  = probe_table.num_rows();
  auto const right_table_row_count = build_table.num_rows();

  std::size_t left_join_complement_size;

  if (left_table_row_count == 0) {
    left_join_complement_size = right_table_row_count;
  } else {
    auto invalid_index_map =
      std::make_unique<rmm::device_uvector<size_type>>(right_table_row_count, stream);
    thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                               invalid_index_map->begin(),
                               invalid_index_map->end(),
                               int32_t{1});

    valid_range<size_type> valid(0, right_table_row_count);

    thrust::scatter_if(rmm::exec_policy_nosync(stream),
                       cuda::make_constant_iterator(0),
                       cuda::make_constant_iterator(0) + right_indices->size(),
                       right_indices->begin(),
                       right_indices->begin(),
                       invalid_index_map->begin(),
                       valid);

    left_join_complement_size = cudf::detail::count_if(
      invalid_index_map->begin(), invalid_index_map->end(), cuda::std::identity{}, stream);
  }
  return join_size + left_join_complement_size;
}

template <typename Hasher>
std::size_t hash_join<Hasher>::full_join_size(cudf::table_view const& probe,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  if (_is_empty) { return probe.num_rows(); }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe, stream);

  return cudf::detail::get_full_join_size(_build,
                                          probe,
                                          _preprocessed_build,
                                          preprocessed_probe,
                                          _hash_table,
                                          _has_nulls,
                                          _nulls_equal,
                                          stream,
                                          mr);
}

template std::size_t hash_join<hash_join_hasher>::full_join_size(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const;

}  // namespace cudf::detail
