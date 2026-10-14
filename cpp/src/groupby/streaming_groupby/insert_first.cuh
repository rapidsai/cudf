/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/for_each.h>

#include <cstddef>
#include <cstring>

namespace cudf::groupby {

/**
 * @brief Invokes a row-equality comparator stored in device memory indirectly.
 *
 * Prevents the underlying comparator's template graph from being inlined into callers.
 *
 * @tparam RowEquality Device-callable row-equality comparator type
 */
template <typename RowEquality>
struct indirect_row_equality {
  RowEquality const* row_equal;  ///< Non-owning device pointer to the row-equality comparator

  __attribute__((noinline)) __device__ bool operator()(size_type lhs, size_type rhs) const noexcept
  {
    return (*row_equal)(lhs, rhs);
  }
};

template <bool has_nested>
size_type streaming_groupby::impl::probe_and_insert_first_batch(
  std::shared_ptr<cudf::detail::row::hash::preprocessed_table> const& preprocessed_batch,
  cudf::nullate::DYNAMIC has_null,
  bitmask_type const* batch_bitmask,
  size_type batch_size,
  size_type* target_indices,
  size_type* batch_local_indices,
  cuda::stream_ref stream)
{
  auto const temp_mr        = cudf::get_current_device_resource_ref();
  auto const batch_self_cmp = cudf::detail::row::equality::self_comparator{preprocessed_batch};
  auto batch_self_eq        = batch_self_cmp.equal_to<has_nested>(has_null, null_equality::EQUAL);

  // This device buffer is intentional: invoking the primitive row comparator indirectly prevents
  // NVCC from inlining its expensive template graph into the CUB kernel, reducing build time and
  // binary size.
  using batch_self_eq_t = decltype(batch_self_eq);
  auto h_batch_self_eq  = cudf::detail::make_empty_pinned_vector<batch_self_eq_t>(1, stream);
  h_batch_self_eq.push_back(batch_self_eq);
  cuda::device_buffer<batch_self_eq_t> d_batch_self_eq(stream, temp_mr, 1, cuda::no_init);
  auto* const d_batch_self_eq_ptr                                   = d_batch_self_eq.data();
  cudf::host_span<batch_self_eq_t const> const h_batch_self_eq_span = h_batch_self_eq;
  cudf::detail::cuda_memcpy_async(
    cudf::device_span<batch_self_eq_t>{d_batch_self_eq.data(), 1}, h_batch_self_eq_span, stream);
  auto const batch_hasher    = cudf::detail::row::hash::row_hasher{preprocessed_batch};
  auto const hasher          = batch_hasher.device_hasher(has_null);
  auto const capacity        = static_cast<cuda::std::uint32_t>(_key_set->size());
  auto const set             = detail::hash::hash_set_ref{_key_set->data(), capacity, capacity};
  auto const first_batch_cmp = first_batch_comparator{
    indirect_row_equality<decltype(batch_self_eq)>{d_batch_self_eq_ptr}, _max_distinct_keys};
  auto* const base = _key_set->data();

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, temp_mr),
    cuda::counting_iterator<size_type>(0),
    batch_size,
    insert_fn{set, first_batch_cmp, hasher, batch_bitmask, _max_distinct_keys, target_indices});
  auto const out_end = thrust::copy_if(rmm::exec_policy_nosync(stream, temp_mr),
                                       cuda::counting_iterator<size_type>(0),
                                       cuda::counting_iterator<size_type>(batch_size),
                                       batch_local_indices,
                                       is_new_key_fn{target_indices, base, _max_distinct_keys});
  return static_cast<size_type>(out_end - batch_local_indices);
}

}  // namespace cudf::groupby
