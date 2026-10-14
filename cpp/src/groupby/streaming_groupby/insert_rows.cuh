/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/for_each.h>

namespace cudf::groupby {

template <bool has_nested>
size_type streaming_groupby::impl::probe_and_insert_rows(
  std::shared_ptr<cudf::detail::row::hash::preprocessed_table> const& preprocessed_batch,
  cudf::nullate::DYNAMIC has_null,
  bitmask_type const* batch_bitmask,
  size_type batch_size,
  size_type* target_indices,
  size_type* batch_local_indices,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  auto const temp_mr        = mr.get_temporary_mr();
  auto const batch_self_cmp = cudf::detail::row::equality::self_comparator{preprocessed_batch};
  auto const batch_self_eq  = batch_self_cmp.equal_to<has_nested>(has_null, null_equality::EQUAL);
  auto const batch_hasher   = cudf::detail::row::hash::row_hasher{preprocessed_batch};
  auto const hasher         = batch_hasher.device_hasher(has_null);
  auto const capacity       = static_cast<cuda::std::uint32_t>(_key_set->size());
  auto const set            = detail::hash::hash_set_ref{_key_set->data(), capacity, capacity};
  auto* const base          = _key_set->data();

  auto d_cross_eqs = build_cross_comparators<has_nested>(
    preprocessed_batch, _preprocessed_batches, has_null, stream, mr);
  auto const comparator = n_table_comparator{
    batch_self_eq, d_cross_eqs.data(), _key_batch_ids->data(), _max_distinct_keys};

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, temp_mr),
    cuda::counting_iterator<size_type>(0),
    batch_size,
    insert_fn{set, comparator, hasher, batch_bitmask, _max_distinct_keys, target_indices});
  auto const out_end = thrust::copy_if(rmm::exec_policy_nosync(stream, temp_mr),
                                       cuda::counting_iterator<size_type>(0),
                                       cuda::counting_iterator<size_type>(batch_size),
                                       batch_local_indices,
                                       is_new_key_fn{target_indices, base, _max_distinct_keys});
  return static_cast<size_type>(out_end - batch_local_indices);
}

}  // namespace cudf::groupby
