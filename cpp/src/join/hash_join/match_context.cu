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
#include <thrust/iterator/transform_output_iterator.h>

namespace cudf::detail {

namespace {
/// Functor that ensures a minimum count of 1 for LEFT/FULL join match counts.
struct clamp_zero_to_one {
  __device__ size_type operator()(size_type count) const { return count == 0 ? 1 : count; }
};
}  // namespace

std::unique_ptr<rmm::device_uvector<size_type>> make_join_match_counts(
  table_view const& right,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_right,
  cudf::detail::hash_table_t const& hash_table,
  bool is_empty,
  bool has_nulls,
  null_equality compare_nulls,
  join_kind join,
  table_view const& left,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto match_counts = std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);

  if (is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 match_counts->begin(),
                 match_counts->end(),
                 join == join_kind::INNER_JOIN ? 0 : 1);
    return match_counts;
  }

  CUDF_EXPECTS(has_nulls || !cudf::has_nested_nulls(left),
               "Left table has nulls while right table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left, stream);
  auto const left_table_num_rows = left.num_rows();

  auto count_matches = [&](auto equality, auto d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});
    if (join == join_kind::INNER_JOIN) {
      hash_table.count_each(iter,
                            iter + left_table_num_rows,
                            equality,
                            hash_table.hash_function(),
                            match_counts->begin(),
                            stream.value());
    } else {
      // For LEFT/FULL joins, fuse the clamp into the output to avoid a separate kernel launch.
      auto const output =
        thrust::make_transform_output_iterator(match_counts->begin(), clamp_zero_to_one{});
      hash_table.count_each(iter,
                            iter + left_table_num_rows,
                            equality,
                            hash_table.hash_function(),
                            output,
                            stream.value());
    }
  };

  dispatch_join_comparator(
    right, left, preprocessed_right, preprocessed_left, has_nulls, compare_nulls, count_matches);

  return match_counts;
}

template <typename Hasher>
std::unique_ptr<rmm::device_uvector<size_type>> hash_join<Hasher>::make_match_counts(
  join_kind join,
  cudf::table_view const& left,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return make_join_match_counts(_right,
                                _preprocessed_right,
                                _impl->_hash_table,
                                _is_empty,
                                _has_nulls,
                                _nulls_equal,
                                join,
                                left,
                                stream,
                                mr);
}

template std::unique_ptr<rmm::device_uvector<size_type>>
hash_join<hash_join_hasher>::make_match_counts(join_kind,
                                               cudf::table_view const&,
                                               rmm::cuda_stream_view,
                                               rmm::device_async_resource_ref) const;

}  // namespace cudf::detail
