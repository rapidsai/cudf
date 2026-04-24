/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "groupby/hash/single_pass_functors.cuh"

#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/for_each.h>

#include <string>

namespace cudf::groupby {

void streaming_groupby::impl::do_aggregate(table_view const& data, rmm::cuda_stream_view stream)
{
  auto const batch_size = data.num_rows();
  if (batch_size == 0) { return; }

  CUDF_EXPECTS(batch_size <= _max_groups,
               "Batch size (" + std::to_string(batch_size) + ") exceeds max_groups (" +
                 std::to_string(_max_groups) + ").",
               std::invalid_argument);

  if (!_initialized) { initialize(data, stream); }

  auto const batch_keys = data.select(_key_indices);

  update_nullable_state(batch_keys);

  if (!_key_set) { create_key_set(stream); }

  auto result = probe_and_insert(batch_keys, stream);

  auto const values_view = data.select(_value_col_indices);
  auto const d_values    = table_device_view::create(values_view, stream);
  auto d_results_ptr     = mutable_table_device_view::create(*_agg_results, stream);

  auto const temp_mr      = cudf::get_current_device_resource_ref();
  auto const num_agg_cols = static_cast<int64_t>(_agg_kinds.size());
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, temp_mr),
    cuda::counting_iterator<int64_t>(0),
    static_cast<int64_t>(batch_size) * num_agg_cols,
    detail::hash::compute_single_pass_aggs_dense_output_fn{
      result.target_indices.begin(), _d_agg_kinds.data(), *d_values, *d_results_ptr});
}

}  // namespace cudf::groupby
