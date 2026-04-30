/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"

#include <cudf/detail/gather.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>

#include <string>

namespace cudf::groupby {

template <bool has_nested>
streaming_groupby::impl::batch_insert_result streaming_groupby::impl::probe_and_insert_impl(
  table_view const& batch_keys, rmm::cuda_stream_view stream)
{
  auto const batch_size = batch_keys.num_rows();
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto const has_null   = cudf::nullate::DYNAMIC{_has_nullable_keys};

  // Preprocess batch for row operators.
  auto preprocessed_batch = cudf::detail::row::hash::preprocessed_table::create(batch_keys, stream);
  auto const batch_hasher_obj = cudf::detail::row::hash::row_hasher{preprocessed_batch};
  auto const d_batch_hash     = batch_hasher_obj.device_hasher(has_null);

  // Compute the null-exclusion bitmask first so the hash cache pass can skip
  // hashing rows that will be excluded — the dummy hash for excluded rows is
  // never read by cuco because `insert_and_map_fn` short-circuits on those rows.
  auto const skip_rows_with_nulls = _has_nullable_keys && _null_handling == null_policy::EXCLUDE;
  auto [bitmask_buffer, batch_bitmask] =
    skip_rows_with_nulls
      ? detail::compute_row_bitmask(batch_keys, stream)
      : std::pair<rmm::device_buffer, bitmask_type const*>{rmm::device_buffer{0, stream}, nullptr};

  // Precompute batch hash values.  Caching is faster than inlining the row hasher
  // inside cuco's probe at high cardinality.
  rmm::device_uvector<hash_value_type> batch_hash_cache(batch_size, stream, temp_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                    cuda::counting_iterator<size_type>(0),
                    cuda::counting_iterator<size_type>(batch_size),
                    batch_hash_cache.begin(),
                    conditional_hash_fn<decltype(d_batch_hash)>{d_batch_hash, batch_bitmask});

  // Build comparator and hasher.
  auto const batch_self_cmp = cudf::detail::row::equality::self_comparator{preprocessed_batch};
  auto const batch_self_eq  = batch_self_cmp.equal_to<has_nested>(has_null, null_equality::EQUAL);

  auto d_cross_eqs = build_cross_comparators<has_nested>(
    preprocessed_batch, _preprocessed_batches, has_null, stream);

  auto const comparator =
    n_table_comparator{batch_self_eq, d_cross_eqs.data(), _key_loc->data(), _max_groups};
  auto const hasher = offset_cache_hasher{batch_hash_cache.data(), _max_groups};

  auto set_ref =
    _key_set->ref(cuco::op::insert_and_find).rebind_key_eq(comparator).rebind_hash_function(hasher);

  // Pass 1 — insert_and_find with transient encoding `_max_groups + batch_idx`.
  // Returns *iter into target_indices and writes inserted_flags + slot_offsets
  // as side outputs.  slot_offsets stores 4-byte slot offsets (vs. 8-byte raw
  // pointers) to halve temp memory.
  auto* const base = _key_set->data();
  rmm::device_uvector<size_type> target_indices(batch_size, stream, temp_mr);
  rmm::device_uvector<bool> inserted_flags(batch_size, stream, temp_mr);
  rmm::device_uvector<size_type> slot_offsets(batch_size, stream, temp_mr);

  thrust::transform(
    rmm::exec_policy_nosync(stream, temp_mr),
    cuda::counting_iterator<size_type>(0),
    cuda::counting_iterator<size_type>(batch_size),
    target_indices.begin(),
    insert_and_map_fn{
      set_ref, batch_bitmask, _max_groups, base, inserted_flags.data(), slot_offsets.data()});

  // Count newly inserted keys.  Sequential reads over inserted_flags are cheaper
  // than the alternative (re-deriving winner status via two random loads per row).
  auto const new_distinct_count = static_cast<size_type>(thrust::count(
    rmm::exec_policy_nosync(stream, temp_mr), inserted_flags.begin(), inserted_flags.end(), true));

  if (new_distinct_count > 0) {
    // Stream compact: get the batch-local indices of newly inserted keys.
    rmm::device_uvector<size_type> batch_local_indices(new_distinct_count, stream, temp_mr);
    thrust::copy_if(rmm::exec_policy_nosync(stream, temp_mr),
                    cuda::counting_iterator<size_type>(0),
                    cuda::counting_iterator<size_type>(batch_size),
                    inserted_flags.begin(),
                    batch_local_indices.begin(),
                    cuda::std::identity{});

    // Check bounds BEFORE any persistent state mutation.
    CUDF_EXPECTS(_distinct_count + new_distinct_count <= _max_groups,
                 "Distinct key count (" + std::to_string(_distinct_count + new_distinct_count) +
                   ") would exceed max_groups (" + std::to_string(_max_groups) + ").");

    // Gather compacted distinct keys from the batch.
    auto compacted = cudf::detail::gather(batch_keys,
                                          batch_local_indices,
                                          out_of_bounds_policy::DONT_CHECK,
                                          cudf::negative_index_policy::NOT_ALLOWED,
                                          stream,
                                          temp_mr);

    auto preprocessed_compacted =
      cudf::detail::row::hash::preprocessed_table::create(compacted->view(), stream);

    // Store the compacted batch.
    auto const new_batch_id          = static_cast<size_type>(_compacted_batches.size());
    auto const distinct_count_before = _distinct_count;
    _compacted_batches.push_back(std::move(compacted));
    _preprocessed_batches.push_back(preprocessed_compacted);

    // Pass 2 — fused: relaxed atomic store rewrites slots from transient encoding
    //                to dense IDs, and writes the {batch_id, row} companion entry.
    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<size_type>(0),
                       new_distinct_count,
                       finalize_new_key_fn{batch_local_indices.data(),
                                           base,
                                           slot_offsets.data(),
                                           _key_loc->data(),
                                           new_batch_id,
                                           distinct_count_before});

    // Re-read target_indices for rows whose slot held a transient value during Pass 1
    // (winners and intra-batch duplicates of new keys).  After Pass 2 every slot holds
    // its final dense ID.  `transform(counting_iterator, ...)` is used instead of
    // `tabulate` to keep nvcc compile times reasonable (see cudf PR #21793).
    thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(batch_size),
                      target_indices.begin(),
                      read_target_indices_fn{base, slot_offsets.data()});

    _distinct_count += new_distinct_count;
  }
  // If new_distinct_count == 0, target_indices is already final from Pass 1 — every
  // slot held a dense ID at probe time, so *iter was already the correct dense ID.

  return batch_insert_result{
    std::move(target_indices), new_distinct_count, std::move(bitmask_buffer)};
}

}  // namespace cudf::groupby
