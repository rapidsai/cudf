/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

cudf::size_type positive_size_axis(nvbench::state const& state, char const* name)
{
  auto const value = state.get_int64(name);
  CUDF_EXPECTS(value > 0 && value <= std::numeric_limits<cudf::size_type>::max(),
               std::string{name} + " must fit positive cudf::size_type");
  return static_cast<cudf::size_type>(value);
}

void bench_streaming_long_string(nvbench::state& state)
{
  auto const num_rows   = positive_size_axis(state, "num_rows");
  auto const batch_size = positive_size_axis(state, "batch_size");
  auto const key_bytes  = positive_size_axis(state, "key_bytes");
  auto const mode       = state.get_string("key_mode");
  CUDF_EXPECTS(mode == "all_unique" || mode == "repeated_batch",
               "key_mode must be all_unique or repeated_batch");
  auto const groups = mode == "all_unique" ? num_rows : std::min(batch_size, num_rows);
  CUDF_EXPECTS(key_bytes >= static_cast<cudf::size_type>(std::to_string(groups - 1).size()),
               "key_bytes must accommodate the longest decimal key");
  CUDF_EXPECTS(
    static_cast<int64_t>(num_rows) * key_bytes <= std::numeric_limits<cudf::size_type>::max(),
    "This fixture keeps string character offsets within INT32_MAX");
  auto const capacity = std::max(groups, std::min(batch_size, num_rows));
  CUDF_EXPECTS(static_cast<int64_t>(capacity) + std::min(batch_size, num_rows) <=
                 std::numeric_limits<cudf::size_type>::max(),
               "Streaming capacity plus batch size must fit cudf::size_type");

  // Decimal digits never contain '#', so right padding preserves distinct keys exactly.
  // Repeated mode reuses precisely the first batch's set; unique mode introduces only new keys.
  auto integer_keys = cudf::sequence(num_rows, cudf::numeric_scalar<int32_t>{0});
  if (mode == "repeated_batch") {
    integer_keys = cudf::binary_operation(integer_keys->view(),
                                          cudf::numeric_scalar<int32_t>{batch_size},
                                          cudf::binary_operator::PYMOD,
                                          cudf::data_type{cudf::type_id::INT32});
  }
  auto decimal_keys = cudf::strings::from_integers(integer_keys->view());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::strings::pad(cudf::strings_column_view{decimal_keys->view()},
                                       key_bytes,
                                       cudf::strings::side_type::RIGHT,
                                       "#"));
  columns.push_back(
    cudf::sequence(num_rows, cudf::numeric_scalar<int32_t>{1}, cudf::numeric_scalar<int32_t>{0}));
  auto const input = std::make_unique<cudf::table>(std::move(columns));
  integer_keys.reset();
  decimal_keys.reset();
  std::vector<cudf::size_type> slice_indices;
  for (int64_t start = 0; start < num_rows; start += batch_size) {
    slice_indices.push_back(static_cast<cudf::size_type>(start));
    slice_indices.push_back(
      static_cast<cudf::size_type>(std::min<int64_t>(start + batch_size, num_rows)));
  }
  auto const batches = cudf::slice(input->view(), slice_indices);
  std::vector<cudf::groupby::streaming_aggregation_request> requests(1);
  requests.front().column_index = 1;
  requests.front().aggregation  = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  std::vector<cudf::size_type> const key_indices{0};
  auto const stream = cudf::get_default_stream();
  stream.sync();
  state.add_element_count(num_rows);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::groupby::streaming_groupby streaming{key_indices, requests, capacity};
    for (auto const& batch : batches) {
      streaming.aggregate(batch, stream);
    }
    [[maybe_unused]] auto const result = streaming.finalize(stream);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

NVBENCH_BENCH(bench_streaming_long_string)
  .set_name("streaming_long_string")
  .add_int64_axis("num_rows", {4'000'000})
  .add_int64_axis("batch_size", {1'000'000})
  .add_int64_axis("key_bytes", {64, 256})
  .add_string_axis("key_mode", {"all_unique", "repeated_batch"});
