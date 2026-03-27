/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "hash_join_helpers.cuh"

#include <cudf/detail/algorithms/reduce.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

namespace cudf::detail {

std::size_t get_full_join_size(
  cudf::table_view const& build_table,
  cudf::table_view const& probe_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_probe,
  hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::size_t join_size = compute_join_output_size(build_table,
                                                   probe_table,
                                                   preprocessed_build,
                                                   preprocessed_probe,
                                                   hash_table,
                                                   join_kind::LEFT_JOIN,
                                                   has_nulls,
                                                   compare_nulls,
                                                   stream);

  if (join_size == 0) { return join_size; }

  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const probe_nulls = cudf::nullate::DYNAMIC{has_nulls};

  cudf::size_type const probe_table_num_rows = probe_table.num_rows();

  auto const out_build_begin =
    thrust::make_transform_output_iterator(right_indices->begin(), output_fn{});

  if (cudf::detail::is_primitive_row_op_compatible(build_table)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{probe_nulls, preprocessed_probe};
    auto const d_equal  = cudf::detail::row::primitive::row_equality_comparator{
      probe_nulls, preprocessed_probe, preprocessed_build, compare_nulls};
    auto const iter     = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});
    auto const equality = primitive_pair_equal{d_equal};

    hash_table.retrieve_outer(iter,
                              iter + probe_table_num_rows,
                              equality,
                              hash_table.hash_function(),
                              cuda::make_discard_iterator(),
                              out_build_begin,
                              stream.value());
  } else {
    auto const d_hasher =
      cudf::detail::row::hash::row_hasher{preprocessed_probe}.device_hasher(probe_nulls);
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});

    auto const row_comparator =
      cudf::detail::row::equality::two_table_comparator{preprocessed_probe, preprocessed_build};
    auto const comparator_helper = [&](auto d_equal) {
      auto const equality = pair_equal{d_equal};
      hash_table.retrieve_outer(iter,
                                iter + probe_table_num_rows,
                                equality,
                                hash_table.hash_function(),
                                cuda::make_discard_iterator(),
                                out_build_begin,
                                stream.value());
    };
    if (cudf::detail::has_nested_columns(probe_table)) {
      auto const d_equal = row_comparator.equal_to<true>(probe_nulls, compare_nulls);
      comparator_helper(d_equal);
    } else {
      auto const d_equal = row_comparator.equal_to<false>(probe_nulls, compare_nulls);
      comparator_helper(d_equal);
    }
  }

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

}  // namespace cudf::detail
