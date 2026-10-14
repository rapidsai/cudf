/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "groupby/common/utils.hpp"

#include <cudf/detail/gather.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <string>

namespace cudf::groupby {

template <bool has_nested>
streaming_groupby::impl::batch_insert_result streaming_groupby::impl::probe_and_insert_impl(
  table_view const& batch_keys, cuda::stream_ref stream)
{
  auto const batch_size = batch_keys.num_rows();
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto const has_null   = cudf::nullate::DYNAMIC{_has_nullable_keys};

  // Preprocess batch for row operators.
  auto preprocessed_batch =
    cudf::detail::row::hash::preprocessed_table::create(batch_keys, stream, temp_mr);

  // Skip hashing and insertion for rows excluded by the null policy.
  auto const skip_rows_with_nulls = _has_nullable_keys && _null_handling == null_policy::EXCLUDE;
  auto [bitmask_buffer, batch_bitmask] =
    skip_rows_with_nulls
      ? detail::compute_row_bitmask(batch_keys, stream)
      : std::pair<cuda::device_buffer<std::byte>, bitmask_type const*>{
          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream), nullptr};

  // Slots are stored in target_indices until dense IDs have been assigned. Only
  // the representative row list needs separate batch-sized scratch.
  rmm::device_uvector<size_type> target_indices(batch_size, stream, temp_mr);
  rmm::device_uvector<size_type> batch_local_indices(batch_size, stream, temp_mr);

  // First batch has no compacted batches yet, so all slot values are transient (>=
  // _max_distinct_keys) and only the batch-self equality branch of n_table_comparator
  // can fire.  Dispatch to a dedicated helper that uses first_batch_comparator to skip
  // the cross-table dispatch, the cross-comparator build, and the dense-ID branches.
  // The two helpers live in separate TUs to parallelize the heavy row-comparator/thrust template
  // instantiations.
  size_type const new_distinct_keys =
    _compacted_batches.empty()
      ? probe_and_insert_first_batch<has_nested>(preprocessed_batch,
                                                 has_null,
                                                 batch_bitmask,
                                                 batch_size,
                                                 target_indices.data(),
                                                 batch_local_indices.data(),
                                                 stream)
      : probe_and_insert_subsequent<has_nested>(preprocessed_batch,
                                                has_null,
                                                batch_bitmask,
                                                batch_size,
                                                target_indices.data(),
                                                batch_local_indices.data(),
                                                stream);
  batch_local_indices.resize(new_distinct_keys, stream);

  if (new_distinct_keys > 0) {
    // Bound check: the hash set has already been written above (transient slot values),
    // so on failure the object is left invalidated; further aggregate()/merge() calls
    // will throw immediately while finalize() can still recover partial results.
    auto const distinct_so_far = _distinct_keys.load(std::memory_order_relaxed);
    if (distinct_so_far + new_distinct_keys > _max_distinct_keys) {
      _invalidated = true;
      CUDF_FAIL("Distinct key count (" + std::to_string(distinct_so_far + new_distinct_keys) +
                ") would exceed max_distinct_keys (" + std::to_string(_max_distinct_keys) + ").");
    }

    // Allocate before publishing new dense IDs so a failed growth leaves the old result
    // prefix and distinct count valid for recovery through finalize().
    ensure_aggregate_capacity(
      distinct_so_far + new_distinct_keys, stream, cudf::memory_resources{temp_mr, temp_mr});

    // Gather compacted distinct keys from the batch.
    auto compacted = cudf::detail::gather(batch_keys,
                                          batch_local_indices,
                                          out_of_bounds_policy::DONT_CHECK,
                                          cudf::negative_index_policy::NOT_ALLOWED,
                                          stream,
                                          temp_mr);

    auto preprocessed_compacted =
      cudf::detail::row::hash::preprocessed_table::create(compacted->view(), stream, temp_mr);

    // Reserve both vectors before enqueueing/publishing the new persistent group IDs.
    _compacted_batches.reserve(_compacted_batches.size() + 1);
    _preprocessed_batches.reserve(_preprocessed_batches.size() + 1);
    auto const new_batch_id    = static_cast<size_type>(_compacted_batches.size());
    auto const dense_id_offset = distinct_so_far;
    // Pass 2 — fused: rewrites slots from transient encoding to dense IDs and
    // writes the {batch_id, row} companion entry.
    auto* const base = _key_set->data();
    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<size_type>(0),
                       new_distinct_keys,
                       finalize_new_key_fn{batch_local_indices.data(),
                                           base,
                                           target_indices.data(),
                                           _key_loc->data(),
                                           new_batch_id,
                                           dense_id_offset});

    _compacted_batches.push_back(std::move(compacted));
    _preprocessed_batches.push_back(std::move(preprocessed_compacted));
    _distinct_keys.fetch_add(new_distinct_keys, std::memory_order_relaxed);
  }

  // Every valid target currently identifies a slot; resident values are now dense IDs.
  thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                     cuda::counting_iterator<size_type>(0),
                     batch_size,
                     update_transient_target_indices_fn{_key_set->data(), target_indices.data()});

  return batch_insert_result{
    std::move(target_indices), new_distinct_keys, std::move(bitmask_buffer)};
}

}  // namespace cudf::groupby
