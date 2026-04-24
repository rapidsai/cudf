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
  auto const num_stored = _num_stored;
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto const has_null   = cudf::nullate::DYNAMIC{_has_nullable_keys};

  CUDF_EXPECTS(static_cast<int64_t>(num_stored) + batch_size <= _max_groups,
               "Cumulative batch rows (" + std::to_string(num_stored) + " + " +
                 std::to_string(batch_size) + ") exceeds max_groups (" +
                 std::to_string(_max_groups) + "). Use smaller batches or increase max_groups.",
               std::overflow_error);

  // Preprocess batch for row operators.
  auto preprocessed_batch = cudf::detail::row::hash::preprocessed_table::create(batch_keys, stream);
  auto const batch_hasher_obj = cudf::detail::row::hash::row_hasher{preprocessed_batch};
  auto const d_batch_hash     = batch_hasher_obj.device_hasher(has_null);

  // Precompute batch hash values.
  rmm::device_uvector<hash_value_type> batch_hash_cache(batch_size, stream, temp_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                    cuda::counting_iterator<size_type>(0),
                    cuda::counting_iterator<size_type>(batch_size),
                    batch_hash_cache.begin(),
                    d_batch_hash);

  // Batch-local bitmask for null exclusion.
  auto const skip_rows_with_nulls = _has_nullable_keys && _null_handling == null_policy::EXCLUDE;
  auto [bitmask_buffer, batch_bitmask] =
    skip_rows_with_nulls
      ? detail::compute_row_bitmask(batch_keys, stream)
      : std::pair<rmm::device_buffer, bitmask_type const*>{rmm::device_buffer{0, stream}, nullptr};

  // Build comparator and hasher.
  auto const batch_self_cmp = cudf::detail::row::equality::self_comparator{preprocessed_batch};
  auto const batch_self_eq  = batch_self_cmp.equal_to<has_nested>(has_null, null_equality::EQUAL);

  auto d_cross_eqs = build_cross_comparators<has_nested>(
    preprocessed_batch, _preprocessed_batches, has_null, stream);

  auto const comparator = n_table_comparator{
    batch_self_eq, d_cross_eqs.data(), _key_batch->data(), _key_row->data(), num_stored};
  auto const hasher = offset_cache_hasher{batch_hash_cache.data(), num_stored};

  auto set_ref =
    _key_set->ref(cuco::op::insert_and_find).rebind_key_eq(comparator).rebind_hash_function(hasher);

  // insert_and_find for all batch rows.
  rmm::device_uvector<size_type> target_indices(batch_size, stream, temp_mr);
  rmm::device_uvector<bool> inserted_flags(batch_size, stream, temp_mr);

  thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                    cuda::counting_iterator<size_type>(0),
                    cuda::counting_iterator<size_type>(batch_size),
                    target_indices.begin(),
                    insert_and_map_fn{set_ref, batch_bitmask, num_stored, inserted_flags.data()});

  // Count newly inserted keys.
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
    auto const new_batch_id = static_cast<size_type>(_compacted_batches.size());
    _compacted_batches.push_back(std::move(compacted));
    _preprocessed_batches.push_back(preprocessed_compacted);

    // Scatter companion vectors and record encoded indices in a single pass.
    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<size_type>(0),
                       new_distinct_count,
                       scatter_new_key_metadata{batch_local_indices.data(),
                                                _key_batch->data(),
                                                _key_row->data(),
                                                _encoded_indices->data(),
                                                num_stored,
                                                new_batch_id,
                                                _distinct_count});

    _distinct_count += new_distinct_count;
  }

  // Advance encoding offset by the full batch size.
  _num_stored += batch_size;

  return batch_insert_result{
    std::move(target_indices), new_distinct_count, std::move(bitmask_buffer)};
}

}  // namespace cudf::groupby
