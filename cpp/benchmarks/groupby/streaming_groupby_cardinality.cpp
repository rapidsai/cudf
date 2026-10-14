/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

cudf::size_type positive_size_axis(nvbench::state const& state, char const* axis)
{
  auto const value = state.get_int64(axis);
  CUDF_EXPECTS(value > 0 && value <= std::numeric_limits<cudf::size_type>::max(),
               std::string{axis} + " must fit positive cudf::size_type");
  return static_cast<cudf::size_type>(value);
}

std::unique_ptr<cudf::table> make_inputs(cudf::size_type num_rows,
                                         cudf::size_type cardinality,
                                         cudf::size_type key_modulus,
                                         std::string const& key_format,
                                         double null_probability,
                                         unsigned seed)
{
  data_profile const key_profile =
    data_profile_builder().cardinality(0).avg_run_length(1).no_validity().distribution(
      cudf::type_id::INT32, distribution_id::UNIFORM, 0, cardinality - 1);
  auto const groups =
    create_random_column(cudf::type_id::INT32, row_count{num_rows}, key_profile, seed);

  auto const key_type = cudf::data_type{cudf::type_id::INT32};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::binary_operation(groups->view(),
                                           cudf::numeric_scalar<int32_t>{key_modulus},
                                           cudf::binary_operator::DIV,
                                           key_type));
  columns.push_back(cudf::binary_operation(groups->view(),
                                           cudf::numeric_scalar<int32_t>{key_modulus},
                                           cudf::binary_operator::PYMOD,
                                           key_type));
  auto const nulls = null_probability > 0 ? std::optional<double>{null_probability} : std::nullopt;
  // Attach independent masks after optional string conversion.
  for (unsigned column = 0; column < 2; ++column) {
    if (key_format == "string") {
      columns[column] = cudf::strings::from_integers(columns[column]->view());
    }
    auto [mask, null_count] = create_random_null_mask(num_rows, nulls, seed + 17u + 14u * column);
    columns[column]->set_null_mask(std::move(mask), null_count);
  }

  data_profile const double_profile =
    data_profile_builder().cardinality(0).avg_run_length(1).null_probability(nulls).distribution(
      cudf::type_id::FLOAT64, distribution_id::UNIFORM, -1.0, 1.0);
  columns.push_back(
    create_random_column(cudf::type_id::FLOAT64, row_count{num_rows}, double_profile, seed + 43u));
  data_profile const integer_profile =
    data_profile_builder().cardinality(0).avg_run_length(1).null_probability(nulls).distribution(
      cudf::type_id::INT32, distribution_id::UNIFORM, -1000, 1000);
  columns.push_back(
    create_random_column(cudf::type_id::INT32, row_count{num_rows}, integer_profile, seed + 71u));
  columns.push_back(
    create_random_column(cudf::type_id::INT32, row_count{num_rows}, integer_profile, seed + 101u));
  return std::make_unique<cudf::table>(std::move(columns));
}

void bench_streaming_groupby_cardinality(nvbench::state& state)
{
  auto const num_rows         = positive_size_axis(state, "num_rows");
  auto const batch_size       = positive_size_axis(state, "batch_size");
  auto const cardinality      = positive_size_axis(state, "cardinality");
  auto const key_modulus      = positive_size_axis(state, "key_modulus");
  auto const key_format       = state.get_string("key_format");
  auto const null_probability = state.get_float64("null_probability");
  auto const seed_value       = state.get_int64("seed");
  CUDF_EXPECTS(key_format == "int32" || key_format == "string",
               "key_format must be int32 or string");
  CUDF_EXPECTS(std::isfinite(null_probability) && null_probability >= 0 && null_probability <= 1,
               "null_probability must be in [0,1]");
  CUDF_EXPECTS(seed_value >= 0 && seed_value <= std::numeric_limits<unsigned>::max(),
               "seed must fit unsigned");

  // INCLUDE admits four null patterns: both valid, only quotient valid, only remainder
  // valid, and both null. Bound all possible composite keys for off-grid configurations.
  auto const quotient_groups =
    static_cast<int64_t>(cardinality) / key_modulus + (cardinality % key_modulus != 0);
  auto const remainder_groups = std::min(cardinality, key_modulus);
  auto const group_bound      = static_cast<int64_t>(cardinality) +
                           (null_probability > 0 ? quotient_groups + remainder_groups + 1 : 0);
  auto const largest_batch = std::min(batch_size, num_rows);
  auto const capacity = std::max<int64_t>(largest_batch, std::min<int64_t>(num_rows, group_bound));
  CUDF_EXPECTS(capacity + largest_batch <= std::numeric_limits<cudf::size_type>::max(),
               "Streaming capacity plus batch size must fit cudf::size_type");

  auto const input = make_inputs(num_rows,
                                 cardinality,
                                 key_modulus,
                                 key_format,
                                 null_probability,
                                 static_cast<unsigned>(seed_value));
  std::vector<cudf::size_type> slice_indices;
  for (int64_t start = 0; start < num_rows; start += batch_size) {
    slice_indices.push_back(static_cast<cudf::size_type>(start));
    slice_indices.push_back(
      static_cast<cudf::size_type>(std::min<int64_t>(start + batch_size, num_rows)));
  }
  auto const batches = cudf::slice(input->view(), slice_indices);

  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  auto const add_request = [&](cudf::size_type column_index,
                               std::unique_ptr<cudf::groupby_aggregation> aggregation) {
    requests.emplace_back();
    requests.back().column_index = column_index;
    requests.back().aggregation  = std::move(aggregation);
  };
  add_request(2, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  add_request(3, cudf::make_max_aggregation<cudf::groupby_aggregation>());
  add_request(4,
              cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
  std::vector<cudf::size_type> const key_indices{0, 1};
  auto const stream = cudf::get_default_stream();
  stream.sync();  // Finish input construction and its temporary deallocation outside both metrics.
  state.add_element_count(num_rows);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::groupby::streaming_groupby streaming{
      key_indices, requests, static_cast<cudf::size_type>(capacity), cudf::null_policy::INCLUDE};
    for (auto const& batch : batches) {
      streaming.aggregate(batch, stream);
    }
    [[maybe_unused]] auto const result = streaming.finalize(stream);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

NVBENCH_BENCH(bench_streaming_groupby_cardinality)
  .set_name("streaming_groupby_cardinality")
  .add_int64_axis("num_rows", {16'000'000})
  .add_int64_axis("batch_size", {1'000'000})
  .add_int64_axis("cardinality", {257, 65'536})
  .add_string_axis("key_format", {"int32", "string"})
  .add_float64_axis("null_probability", {0.0, 0.1})
  .add_int64_axis("key_modulus", {97})
  .add_int64_axis("seed", {12'345});
