/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "dispatch.cuh"
#include "join/join_common_utils.cuh"
#include "join/join_common_utils.hpp"
#include "retrieve_kernels.hpp"

#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

namespace cudf::detail {
namespace {

/**
 * @brief Returns trivial left/right index pairs for an outer join when the build side is empty.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
make_trivial_outer_indices(size_type left_start_idx,
                           size_type partition_size,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(partition_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(partition_size, stream, mr);
  auto out           = cuda::zip_iterator(left_indices->begin(), right_indices->begin());
  thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   out,
                   out + partition_size,
                   cuda::proclaim_return_type<cuda::std::tuple<size_type, size_type>>(
                     [left_start_idx] __device__(auto i) {
                       return cuda::std::tuple{static_cast<size_type>(left_start_idx + i),
                                               JoinNoMatch};
                     }));
  return std::pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::partitioned_join_retrieve(join_kind join,
                                             cudf::join_partition_context const& context,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(
    join == join_kind::INNER_JOIN || join == join_kind::LEFT_JOIN || join == join_kind::FULL_JOIN,
    "Unsupported join kind for partitioned retrieve");

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
    if (join == join_kind::INNER_JOIN) {
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    } else {
      return make_trivial_outer_indices(left_start_idx, partition_size, stream, mr);
    }
  }

  // Slice the probe table to the partition range
  auto const probe_partition_view =
    cudf::slice(match_ctx._left_table, {left_start_idx, left_end_idx})[0];

  validate_hash_join_probe(_build, probe_partition_view, _has_nulls);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe_partition_view, stream);

  // For FULL_JOIN, probe with LEFT_JOIN semantics (no complement here)
  bool const is_outer = (join != join_kind::INNER_JOIN);

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

    if (is_outer) {
      join_indices = launch_retrieve<true>(
        probe_keys.data(), n, partition_counts, ref, left_start_idx, stream, mr);
    } else {
      join_indices = launch_retrieve<false>(
        probe_keys.data(), n, partition_counts, ref, left_start_idx, stream, mr);
    }
  };

  dispatch_join_comparator(_build,
                           probe_partition_view,
                           _preprocessed_build,
                           preprocessed_probe,
                           _has_nulls,
                           _nulls_equal,
                           retrieve_partition);

  return join_indices;
}

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::partitioned_join_retrieve(join_kind,
                                                       cudf::join_partition_context const&,
                                                       rmm::cuda_stream_view,
                                                       rmm::device_async_resource_ref) const;

}  // namespace cudf::detail
