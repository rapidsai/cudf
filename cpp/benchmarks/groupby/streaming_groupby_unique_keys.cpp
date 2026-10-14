/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

void verify_result(cudf::table_view keys,
                   std::vector<cudf::groupby::aggregation_result> const& results,
                   cudf::size_type groups,
                   bool with_min,
                   cuda::stream_ref stream)
{
  CUDF_EXPECTS(keys.num_columns() == 1 && keys.num_rows() == groups,
               "Streaming benchmark returned an incorrect key shape");
  auto const key_column = keys.column(0);
  CUDF_EXPECTS(key_column.type().id() == cudf::type_id::INT32 && !key_column.has_nulls(),
               "Streaming benchmark returned an incorrect key type or validity");
  CUDF_EXPECTS(results.size() == (with_min ? 1 : 0),
               "Streaming benchmark returned an incorrect request count");
  std::vector<int32_t> host_keys(groups);
  cudf::detail::cuda_memcpy(
    cudf::host_span<int32_t>{host_keys},
    cudf::device_span<int32_t const>{key_column.data<int32_t>(), static_cast<std::size_t>(groups)},
    stream);
  std::vector<int8_t> host_values;
  if (with_min) {
    CUDF_EXPECTS(results.front().results.size() == 1,
                 "Streaming benchmark returned an incorrect aggregation count");
    auto const values = results.front().results.front()->view();
    CUDF_EXPECTS(
      values.type().id() == cudf::type_id::INT8 && values.size() == groups && !values.has_nulls(),
      "Streaming benchmark returned an incorrect MIN shape, type or validity");
    host_values.resize(groups);
    cudf::detail::cuda_memcpy(
      cudf::host_span<int8_t>{host_values},
      cudf::device_span<int8_t const>{values.data<int8_t>(), static_cast<std::size_t>(groups)},
      stream);
  }
  std::vector<uint8_t> seen(groups, 0);
  for (cudf::size_type row = 0; row < groups; ++row) {
    auto const key = host_keys[row];
    CUDF_EXPECTS(key >= 0 && key < groups, "Streaming benchmark returned an out-of-range key");
    CUDF_EXPECTS(seen[key]++ == 0, "Streaming benchmark returned a duplicate key");
    if (with_min) {
      CUDF_EXPECTS(host_values[row] == static_cast<int8_t>(key % 127 - 63),
                   "Streaming benchmark returned an incorrect MIN value");
    }
  }
}

void bench_streaming_groupby_unique_keys(nvbench::state& state)
{
  auto const group_axis = state.get_int64("max_distinct_keys");
  auto const batch_axis = state.get_int64("batch_size");
  auto const schema     = state.get_string("schema");
  CUDF_EXPECTS(group_axis > 0 && group_axis <= std::numeric_limits<cudf::size_type>::max() &&
                 batch_axis > 0 && batch_axis <= std::numeric_limits<cudf::size_type>::max() &&
                 batch_axis <= group_axis &&
                 group_axis + batch_axis <= std::numeric_limits<cudf::size_type>::max(),
               "Capacity and batch size must fit the streaming row-index bound");
  CUDF_EXPECTS(schema == "keys_only" || schema == "int8_min", "Unsupported schema");
  auto const groups     = static_cast<cudf::size_type>(group_axis);
  auto const batch_size = static_cast<cudf::size_type>(batch_axis);
  auto const with_min   = schema == "int8_min";
  auto const stream     = cudf::get_default_stream();

  // Unique keys arrive in small batches, so the number of groups grows throughout the operation.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::sequence(groups, cudf::numeric_scalar<int32_t>{0}));
  if (with_min) {
    auto const residues = cudf::binary_operation(columns.front()->view(),
                                                 cudf::numeric_scalar<int32_t>{127},
                                                 cudf::binary_operator::PYMOD,
                                                 cudf::data_type{cudf::type_id::INT32});
    columns.push_back(cudf::binary_operation(residues->view(),
                                             cudf::numeric_scalar<int32_t>{63},
                                             cudf::binary_operator::SUB,
                                             cudf::data_type{cudf::type_id::INT8}));
  }
  auto const input = std::make_unique<cudf::table>(std::move(columns));
  std::vector<cudf::size_type> slice_indices;
  for (int64_t begin = 0; begin < groups; begin += batch_size) {
    slice_indices.push_back(static_cast<cudf::size_type>(begin));
    slice_indices.push_back(
      static_cast<cudf::size_type>(std::min<int64_t>(begin + batch_size, groups)));
  }
  auto const batches = cudf::slice(input->view(), slice_indices);
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  if (with_min) {
    requests.emplace_back();
    requests.back().column_index = 1;
    requests.back().aggregation  = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  }
  std::vector<cudf::size_type> const key_indices{0};
  auto const run = [&](bool verify) {
    cudf::groupby::streaming_groupby streaming{key_indices, requests, groups};
    for (auto const& batch : batches) {
      streaming.aggregate(batch, stream);
    }
    auto const result = streaming.finalize(stream);
    if (verify) { verify_result(result.first->view(), result.second, groups, with_min, stream); }
    // Both result and object destruction belong to the measured operation.
  };

  // Validate one complete operation outside both metrics. Input/verification owners
  // and their queued deallocations must finish before the memory logger is installed.
  run(true);
  stream.sync();
  state.add_element_count(groups);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) { run(false); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

NVBENCH_BENCH(bench_streaming_groupby_unique_keys)
  .set_name("streaming_groupby_unique_keys")
  .add_int64_axis("max_distinct_keys", {65'537, 1'048'577})
  .add_int64_axis("batch_size", {4096})
  .add_string_axis("schema", {"keys_only", "int8_min"});
