/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "groupby/hash/single_pass_functors.cuh"
#include "partial_aggregates.cuh"
#include "peer_aggregate.hpp"

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>

namespace cudf::groupby {

namespace {

struct assign_partial_targets {
  size_type* targets;
  size_type replicas;
  size_type capacity;

  __device__ void operator()(size_type row) const
  {
    auto const group = targets[row];
    if (group != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
      targets[row] = ((row / partial_aggregation_block_size) % replicas) * capacity + group;
    }
  }
};

}  // namespace

void streaming_groupby::impl::do_aggregate(table_view const& data, cuda::stream_ref stream)
{
  ensure_not_invalidated();

  auto const batch_size = data.num_rows();
  if (batch_size == 0) { return; }

  CUDF_EXPECTS(batch_size <= _max_distinct_keys,
               "Batch size (" + std::to_string(batch_size) + ") exceeds max_distinct_keys (" +
                 std::to_string(_max_distinct_keys) + ").",
               std::invalid_argument);

  CUDF_EXPECTS(static_cast<int64_t>(_max_distinct_keys) + static_cast<int64_t>(batch_size) <=
                 static_cast<int64_t>(std::numeric_limits<size_type>::max()),
               "Transient key encoding (max_distinct_keys + batch_size) would overflow size_type.",
               std::invalid_argument);

  // Keep host bookkeeping locked until every result use is enqueued and registered.
  std::lock_guard const lock{_insert_mutex};
  ensure_not_invalidated();
  bool insertion_started = false;
  try {
    if (!_initialized) { initialize(data, stream); }
    // An initial tiny batch must not permanently limit later batches to one shard. Increasing
    // the budget only after doubling bounds how often stable groups require a reshape.
    auto const batch_budget = std::max<size_type>(1, batch_size / partial_aggregation_block_size);
    if (_partial_budget == 0 || static_cast<int64_t>(batch_budget) >= 2LL * _partial_budget) {
      _partial_budget = batch_budget;
    }
    auto const batch_keys = data.select(_key_indices);
    update_nullable_state(batch_keys);
    if (!_key_set) { create_key_set(stream); }
    _insert_done.wait(stream);
    insertion_started = true;
    auto result       = probe_and_insert(batch_keys, stream);
    _insert_done.record(stream);
    auto const num_groups = _distinct_keys.load(std::memory_order_relaxed);
    auto const temp_mr    = cudf::get_current_device_resource_ref();
    // probe_and_insert handles capacity growth before publishing new groups. This additional
    // check handles a larger batch budget when all keys were already present.
    ensure_aggregate_capacity(num_groups, stream, cudf::memory_resources{temp_mr, temp_mr});
    _results_read_done.wait(stream);

    auto const num_items =
      static_cast<int64_t>(batch_size) * static_cast<int64_t>(_agg_kinds.size());
    if (num_items > 0 && num_groups > 0) {
      if (_aggregate_replicas > 1) {
        thrust::for_each_n(
          rmm::exec_policy_nosync(stream, temp_mr),
          cuda::counting_iterator<size_type>{0},
          batch_size,
          assign_partial_targets{
            result.target_indices.data(), _aggregate_replicas, _aggregate_capacity});
      }
      compute_peer_aggs(data.select(_value_col_indices),
                        _agg_kinds,
                        _d_agg_kinds->data(),
                        {result.target_indices.data(), result.target_indices.size()},
                        _agg_results->mutable_view(),
                        *_d_agg_results,
                        stream,
                        cudf::memory_resources{temp_mr, temp_mr});
    }
    record_aggregate_completion(stream);
  } catch (...) {
    if (insertion_started) { _invalidated = true; }
    // Do not let partially enqueued writes escape tracking before a subsequent resize/read.
    static_cast<void>(cudaStreamSynchronize(stream.get()));
    throw;
  }
}

}  // namespace cudf::groupby
