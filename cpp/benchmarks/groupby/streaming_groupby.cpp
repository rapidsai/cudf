/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <memory>
#include <vector>

static void bench_streaming_groupby(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const batch_size  = static_cast<cudf::size_type>(state.get_int64("batch_size"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const key_order   = state.get_string("key_order");
  auto const aggs        = state.get_string("aggs");
  // Sorted runs of a single row are the same keys as the round robin order.
  if (key_order == "sorted" && batch_size / cardinality <= 1) {
    state.skip("sorted keys equal round robin keys without runs of equal keys");
    return;
  }

  auto const keys = [&] {
    if (key_order == "uniform_random") {
      data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
        cudf::type_id::INT32, distribution_id::UNIFORM, 0, cardinality - 1);
      return create_random_column(cudf::type_id::INT32, row_count{num_rows}, profile);
    }
    auto const int32_type = cudf::data_type{cudf::type_id::INT32};
    auto const sequence   = cudf::sequence(num_rows, cudf::numeric_scalar<int32_t>(0));
    if (key_order == "round_robin") {
      return cudf::binary_operation(sequence->view(),
                                    cudf::numeric_scalar<int32_t>(cardinality),
                                    cudf::binary_operator::PYMOD,
                                    int32_type);
    }
    auto const run  = std::max(1, batch_size / cardinality);
    auto const runs = cudf::binary_operation(
      sequence->view(), cudf::numeric_scalar<int32_t>(run), cudf::binary_operator::DIV, int32_type);
    return cudf::binary_operation(runs->view(),
                                  cudf::numeric_scalar<int32_t>(cardinality),
                                  cudf::binary_operator::PYMOD,
                                  int32_type);
  }();
  data_profile const int_profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_id::INT32, distribution_id::UNIFORM, 0, 100);
  auto const int_vals =
    create_random_column(cudf::type_id::INT32, row_count{num_rows}, int_profile);
  data_profile const double_profile =
    data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_id::FLOAT64, distribution_id::UNIFORM, 0, 100);
  auto const double_vals =
    create_random_column(cudf::type_id::FLOAT64, row_count{num_rows}, double_profile);

  std::vector<cudf::size_type> slice_indices;
  for (cudf::size_type start = 0; start < num_rows; start += batch_size) {
    slice_indices.push_back(start);
    slice_indices.push_back(std::min(start + batch_size, num_rows));
  }
  auto const batches = cudf::slice(
    cudf::table_view({keys->view(), int_vals->view(), double_vals->view()}), slice_indices);

  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  auto const add_request = [&](cudf::size_type column_index,
                               std::unique_ptr<cudf::groupby_aggregation> aggregation) {
    requests.emplace_back();
    requests.back().column_index = column_index;
    requests.back().aggregation  = std::move(aggregation);
  };
  add_request(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  if (aggs == "mixed") {
    add_request(1, cudf::make_min_aggregation<cudf::groupby_aggregation>());
    add_request(2, cudf::make_max_aggregation<cudf::groupby_aggregation>());
    add_request(2, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  }

  std::vector<cudf::size_type> const key_indices{0};
  state.add_element_count(num_rows);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    // Streaming groupby also requires capacity for every row in an individual batch.
    auto sgb =
      cudf::groupby::streaming_groupby(key_indices, requests, std::max(batch_size, cardinality));
    for (auto const& batch : batches) {
      sgb.aggregate(batch);
    }
    auto const result = sgb.finalize();
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_streaming_groupby)
  .set_name("streaming_groupby")
  .add_int64_power_of_two_axis("num_rows", {24})
  .add_int64_power_of_two_axis("batch_size", {10, 14, 20})
  .add_int64_axis("cardinality", {32, 256, 512, 65'536, 1'048'576})
  .add_string_axis("key_order", {"uniform_random", "round_robin", "sorted"})
  .add_string_axis("aggs", {"sum", "mixed"});
