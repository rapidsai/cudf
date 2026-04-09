/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

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

streaming_groupby::impl::pre_insert_result streaming_groupby::impl::pre_insert(
  table_view const& batch_keys, rmm::cuda_stream_view stream)
{
  auto const batch_size = batch_keys.num_rows();
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto const has_null   = cudf::nullate::DYNAMIC{_has_nullable_keys};

  CUDF_EXPECTS(static_cast<int64_t>(_num_stored) + batch_size <= _max_groups,
               "Cumulative batch rows (" + std::to_string(_num_stored) + " + " +
                 std::to_string(batch_size) + ") exceeds max_groups (" +
                 std::to_string(_max_groups) + "). Use smaller batches or increase max_groups.",
               std::overflow_error);

  auto preprocessed_batch = cudf::detail::row::hash::preprocessed_table::create(batch_keys, stream);
  auto const batch_hasher_obj = cudf::detail::row::hash::row_hasher{preprocessed_batch};
  auto const d_batch_hash     = batch_hasher_obj.device_hasher(has_null);

  rmm::device_uvector<hash_value_type> batch_hash_cache(batch_size, stream, temp_mr);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator<size_type>(0),
                    cuda::counting_iterator<size_type>(batch_size),
                    batch_hash_cache.begin(),
                    d_batch_hash);

  auto const skip_rows_with_nulls = _has_nullable_keys && _null_handling == null_policy::EXCLUDE;
  auto [bitmask_buffer, batch_bitmask] =
    skip_rows_with_nulls
      ? detail::compute_row_bitmask(batch_keys, stream)
      : std::pair<rmm::device_buffer, bitmask_type const*>{rmm::device_buffer{0, stream}, nullptr};

  return {std::move(preprocessed_batch),
          std::move(batch_hash_cache),
          std::move(bitmask_buffer),
          batch_bitmask};
}

streaming_groupby::impl::batch_insert_result streaming_groupby::impl::post_insert(
  table_view const& batch_keys,
  rmm::device_uvector<size_type>&& target_indices,
  rmm::device_uvector<bool>& inserted_flags,
  rmm::device_buffer&& bitmask_buffer,
  rmm::cuda_stream_view stream)
{
  auto const batch_size = batch_keys.num_rows();
  auto const num_stored = _num_stored;
  auto const temp_mr    = cudf::get_current_device_resource_ref();

  auto const new_distinct_count = static_cast<size_type>(thrust::count(
    rmm::exec_policy_nosync(stream), inserted_flags.begin(), inserted_flags.end(), true));

  if (new_distinct_count > 0) {
    rmm::device_uvector<size_type> batch_local_indices(new_distinct_count, stream, temp_mr);
    thrust::copy_if(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator<size_type>(0),
                    cuda::counting_iterator<size_type>(batch_size),
                    inserted_flags.begin(),
                    batch_local_indices.begin(),
                    cuda::std::identity{});

    CUDF_EXPECTS(_distinct_count + new_distinct_count <= _max_groups,
                 "Distinct key count (" + std::to_string(_distinct_count + new_distinct_count) +
                   ") would exceed max_groups (" + std::to_string(_max_groups) + ").");

    auto compacted = cudf::detail::gather(batch_keys,
                                          batch_local_indices,
                                          out_of_bounds_policy::DONT_CHECK,
                                          cudf::negative_index_policy::NOT_ALLOWED,
                                          stream,
                                          temp_mr);

    auto preprocessed_compacted =
      cudf::detail::row::hash::preprocessed_table::create(compacted->view(), stream);

    auto const new_batch_id = static_cast<size_type>(_compacted_batches.size());
    _compacted_batches.push_back(std::move(compacted));
    _preprocessed_batches.push_back(preprocessed_compacted);

    thrust::for_each_n(rmm::exec_policy_nosync(stream),
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

  _num_stored += batch_size;

  return batch_insert_result{
    std::move(target_indices), new_distinct_count, std::move(bitmask_buffer)};
}

streaming_groupby::impl::batch_insert_result streaming_groupby::impl::probe_and_insert(
  table_view const& batch_keys, rmm::cuda_stream_view stream)
{
  auto pre = pre_insert(batch_keys, stream);

  auto [target_indices, inserted_flags] = _has_nested_keys
                                            ? do_insert<true>(batch_keys, pre, stream)
                                            : do_insert<false>(batch_keys, pre, stream);

  return post_insert(
    batch_keys, std::move(target_indices), inserted_flags, std::move(pre.bitmask_buffer), stream);
}

}  // namespace cudf::groupby
