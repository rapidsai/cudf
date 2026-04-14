/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common.cuh"
#include "dispatch.cuh"
#include "join/join_common_utils.cuh"
#include "join/join_common_utils.hpp"
#include "retrieve_kernels.hpp"
#include "size_impl.cuh"

#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <thrust/fill.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace cudf::detail {

template <join_kind Join>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
probe_join_hash_table(
  cudf::table_view const& build_table,
  cudf::table_view const& probe_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_probe,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  std::optional<std::size_t> output_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static_assert(Join == join_kind::INNER_JOIN || Join == join_kind::LEFT_JOIN ||
                Join == join_kind::FULL_JOIN);

  constexpr auto size_join = Join == join_kind::FULL_JOIN ? join_kind::LEFT_JOIN : Join;

  std::size_t const join_size = output_size
                                  ? *output_size
                                  : compute_join_output_size<size_join>(build_table,
                                                                        probe_table,
                                                                        preprocessed_build,
                                                                        preprocessed_probe,
                                                                        hash_table,
                                                                        has_nulls,
                                                                        compare_nulls,
                                                                        stream);

  if (join_size == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  cudf::prefetch::detail::prefetch(*left_indices, stream);
  cudf::prefetch::detail::prefetch(*right_indices, stream);

  auto const probe_table_num_rows = probe_table.num_rows();
  auto const out_probe_begin =
    thrust::make_transform_output_iterator(left_indices->begin(), output_fn{});
  auto const out_build_begin =
    thrust::make_transform_output_iterator(right_indices->begin(), output_fn{});

  auto retrieve_results = [&](auto equality, auto d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});
    if constexpr (Join == join_kind::INNER_JOIN) {
      hash_table.retrieve(iter,
                          iter + probe_table_num_rows,
                          equality,
                          hash_table.hash_function(),
                          out_probe_begin,
                          out_build_begin,
                          stream.value());
    } else {
      [[maybe_unused]] auto out_probe_end = hash_table
                                              .retrieve_outer(iter,
                                                              iter + probe_table_num_rows,
                                                              equality,
                                                              hash_table.hash_function(),
                                                              out_probe_begin,
                                                              out_build_begin,
                                                              stream.value())
                                              .first;

      if constexpr (Join == join_kind::FULL_JOIN) {
        auto const actual_size = cuda::std::distance(out_probe_begin, out_probe_end);
        left_indices->resize(actual_size, stream);
        right_indices->resize(actual_size, stream);
      }
    }
  };

  dispatch_join_comparator(build_table,
                           probe_table,
                           preprocessed_build,
                           preprocessed_probe,
                           has_nulls,
                           compare_nulls,
                           retrieve_results);

  return std::pair(std::move(left_indices), std::move(right_indices));
}

template <typename RightOutputIterator>
void retrieve_left_join_build_indices(
  cudf::table_view const& build_table,
  cudf::table_view const& probe_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_probe,
  cudf::detail::hash_table_t const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  RightOutputIterator out_build_begin,
  rmm::cuda_stream_view stream)
{
  auto const probe_table_num_rows = probe_table.num_rows();

  auto retrieve_results = [&](auto equality, auto d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});
    hash_table.retrieve_outer(iter,
                              iter + probe_table_num_rows,
                              equality,
                              hash_table.hash_function(),
                              cuda::make_discard_iterator(),
                              out_build_begin,
                              stream.value());
  };

  dispatch_join_comparator(build_table,
                           probe_table,
                           preprocessed_build,
                           preprocessed_probe,
                           has_nulls,
                           compare_nulls,
                           retrieve_results);
}

template <typename Hasher>
template <join_kind Join>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::join_retrieve(cudf::table_view const& probe,
                                 std::optional<std::size_t> output_size,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  validate_hash_join_probe(_build, probe, _has_nulls);

  if constexpr (Join == join_kind::INNER_JOIN) {
    if (is_trivial_join(probe, _build, Join)) {
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }
  } else {
    if (_is_empty) { return get_trivial_left_join_indices(probe, stream, mr); }

    if (is_trivial_join(probe, _build, Join)) {
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }
  }

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe, stream);

  auto join_indices = cudf::detail::probe_join_hash_table<Join>(_build,
                                                                probe,
                                                                _preprocessed_build,
                                                                preprocessed_probe,
                                                                _impl->_hash_table,
                                                                _has_nulls,
                                                                _nulls_equal,
                                                                output_size,
                                                                stream,
                                                                mr);

  if constexpr (Join == join_kind::FULL_JOIN) {
    auto complement_indices = detail::get_left_join_indices_complement(
      join_indices.second, probe.num_rows(), _build.num_rows(), stream, mr);
    return detail::concatenate_vector_pairs(join_indices, complement_indices, stream);
  } else {
    return join_indices;
  }
}

template <typename Hasher>
template <join_kind Join>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::partitioned_join_retrieve(cudf::join_partition_context const& context,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  static_assert(Join == join_kind::INNER_JOIN || Join == join_kind::LEFT_JOIN ||
                Join == join_kind::FULL_JOIN);

  auto const& match_ctx     = *context.left_table_context;
  auto const left_start_idx = context.left_start_idx;
  auto const left_end_idx   = context.left_end_idx;

  // Empty partition
  if (left_start_idx >= left_end_idx) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto const partition_size = left_end_idx - left_start_idx;

  // Trivial case: build table is empty
  if (_is_empty) {
    if constexpr (Join == join_kind::INNER_JOIN) {
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    } else {
      auto left_indices =
        std::make_unique<rmm::device_uvector<size_type>>(partition_size, stream, mr);
      auto right_indices =
        std::make_unique<rmm::device_uvector<size_type>>(partition_size, stream, mr);
      thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       left_indices->begin(),
                       left_indices->end(),
                       left_start_idx);
      thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   right_indices->begin(),
                   right_indices->end(),
                   JoinNoMatch);
      return std::pair(std::move(left_indices), std::move(right_indices));
    }
  }

  // Slice the probe table to the partition range
  auto const probe_partition_view =
    cudf::slice(match_ctx._left_table, {left_start_idx, left_end_idx})[0];

  validate_hash_join_probe(_build, probe_partition_view, _has_nulls);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe_partition_view, stream);

  // For FULL_JOIN, probe with LEFT_JOIN semantics (no complement here)
  constexpr bool is_outer = (Join != join_kind::INNER_JOIN);

  // launch_retrieve computes output size from match counts via exclusive scan
  // (total = last_offset + last_count), allocates output buffers, and launches the kernel.
  auto const* partition_counts = match_ctx._match_counts->data() + left_start_idx;
  auto const n                 = static_cast<cuda::std::int64_t>(partition_size);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
    join_indices;

  auto retrieve_partition = [&](auto equality, auto d_hasher) {
    // Precompute probe keys for this partition slice.
    rmm::device_uvector<probe_key_type> probe_keys(n, stream);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(partition_size),
                      probe_keys.begin(),
                      pair_fn{d_hasher});

    auto const ref = _impl->_hash_table.ref(cuco::op::count)
                       .rebind_key_eq(equality)
                       .rebind_hash_function(_impl->_hash_table.hash_function());

    join_indices =
      launch_retrieve<is_outer>(probe_keys.data(), n, partition_counts, ref, stream, mr);
  };

  dispatch_join_comparator(_build,
                           probe_partition_view,
                           _preprocessed_build,
                           preprocessed_probe,
                           _has_nulls,
                           _nulls_equal,
                           retrieve_partition);

  auto& [left_indices, right_indices] = join_indices;

  // Offset left indices to be relative to the original complete probe table
  if (left_start_idx > 0 && left_indices->size() > 0) {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      left_indices->begin(),
                      left_indices->end(),
                      cuda::make_constant_iterator(left_start_idx),
                      left_indices->begin(),
                      cuda::std::plus<size_type>{});
  }

  return join_indices;
}

}  // namespace cudf::detail
