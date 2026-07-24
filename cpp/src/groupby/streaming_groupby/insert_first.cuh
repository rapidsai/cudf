/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"

#include <cudf/detail/device_scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cub/device/dispatch/dispatch_select_if.cuh>
#include <cuda/iterator>

namespace cudf::groupby {

// Limit the number of copies of the state-heavy insert predicate in each CUB agent. This retains
// stable, single-pass selection while reducing register pressure and device-code optimization time.
struct first_batch_select_policy {
  struct Policy900 : cub::ChainedPolicy<900, Policy900, Policy900> {
    using SelectIfPolicyT = cub::AgentSelectIfPolicy<128,
                                                     5,
                                                     cub::BLOCK_LOAD_DIRECT,
                                                     cub::LOAD_DEFAULT,
                                                     cub::BLOCK_SCAN_WARP_SCANS,
                                                     cub::detail::no_delay_constructor_t<0>>;
  };

  using MaxPolicy = Policy900;
};

template <bool has_nested>
size_type streaming_groupby::impl::probe_and_insert_first_batch(
  std::shared_ptr<cudf::detail::row::hash::preprocessed_table> const& preprocessed_batch,
  cudf::nullate::DYNAMIC has_null,
  bitmask_type const* batch_bitmask,
  hash_value_type const* batch_hash_cache,
  size_type batch_size,
  size_type* target_indices,
  size_type* slot_offsets,
  size_type* batch_local_indices,
  rmm::cuda_stream_view stream)
{
  auto const temp_mr        = cudf::get_current_device_resource_ref();
  auto const batch_self_cmp = cudf::detail::row::equality::self_comparator{preprocessed_batch};
  auto const batch_self_eq  = batch_self_cmp.equal_to<has_nested>(has_null, null_equality::EQUAL);
  auto const hasher         = offset_cache_hasher{batch_hash_cache, _max_distinct_keys};
  auto const set_ref_base   = _key_set->ref(cuco::op::insert_and_find).rebind_hash_function(hasher);
  auto const first_batch_cmp = first_batch_comparator{batch_self_eq, _max_distinct_keys};
  auto* const base           = _key_set->data();
  auto const input           = cuda::counting_iterator<size_type>(0);
  auto const predicate       = insert_and_check_fn{set_ref_base.rebind_key_eq(first_batch_cmp),
                                             batch_bitmask,
                                             _max_distinct_keys,
                                             base,
                                             target_indices,
                                             slot_offsets};
  cudf::detail::device_scalar<size_type> output_count(0, stream, temp_mr);

  using dispatch_t = cub::DispatchSelectIf<decltype(input),
                                           cub::NullType*,
                                           size_type*,
                                           size_type*,
                                           decltype(predicate),
                                           cub::NullType,
                                           size_type,
                                           cub::SelectImpl::Select,
                                           first_batch_select_policy>;
  std::size_t temporary_storage_bytes{};
  CUDF_CUDA_TRY(dispatch_t::Dispatch(nullptr,
                                     temporary_storage_bytes,
                                     input,
                                     nullptr,
                                     batch_local_indices,
                                     output_count.data(),
                                     predicate,
                                     cub::NullType{},
                                     batch_size,
                                     stream.value()));
  rmm::device_buffer temporary_storage(temporary_storage_bytes, stream, temp_mr);
  CUDF_CUDA_TRY(dispatch_t::Dispatch(temporary_storage.data(),
                                     temporary_storage_bytes,
                                     input,
                                     nullptr,
                                     batch_local_indices,
                                     output_count.data(),
                                     predicate,
                                     cub::NullType{},
                                     batch_size,
                                     stream.value()));
  return output_count.value(stream);
}

}  // namespace cudf::groupby
