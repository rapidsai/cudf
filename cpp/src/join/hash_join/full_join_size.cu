/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "retrieve_impl.cuh"

#include <cudf/detail/algorithms/reduce.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/iterator/transform_output_iterator.h>

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

  return join_size + compute_left_join_complement_size(
                       right_indices, left_table_row_count, right_table_row_count, stream);
}

template <typename Hasher>
std::size_t hash_join<Hasher>::full_join_size(cudf::table_view const& probe,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr) const
{
  return this->template join_size<join_kind::FULL_JOIN>(probe, stream, mr);
}

template std::size_t hash_join<hash_join_hasher>::full_join_size(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const;

}  // namespace cudf::detail
