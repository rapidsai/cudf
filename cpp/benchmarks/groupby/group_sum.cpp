/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>

#include <nvbench/nvbench.cuh>

using Types = nvbench::type_list<int64_t, numeric::decimal64>;
NVBENCH_DECLARE_TYPE_STRINGS(numeric::decimal64, "decimal64", "decimal64");

template <typename DataType>
static void bench_groupby_basic_sum(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const data_type_id = cudf::type_to_id<DataType>();

  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    data_type_id, distribution_id::UNIFORM, 0, 100);
  auto keys = create_random_column(data_type_id, row_count{num_rows}, profile);
  auto vals = create_random_column(data_type_id, row_count{num_rows}, profile);

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = vals->view();
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  state.add_global_memory_reads<nvbench::int8_t>(vals->alloc_size());
  std::size_t write_size = 0;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::groupby::groupby gb_obj(cudf::table_view({keys->view(), keys->view(), keys->view()}));
    auto const result = gb_obj.aggregate(requests);
    write_size = result.first->alloc_size() + result.second.front().results.front()->alloc_size();
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  state.add_global_memory_writes<nvbench::int8_t>(write_size);
}

NVBENCH_BENCH_TYPES(bench_groupby_basic_sum, NVBENCH_TYPE_AXES(Types))
  .set_name("sum")
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000});

template <typename DataType>
static void bench_groupby_pre_sorted_sum(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const data_type_id = cudf::type_to_id<DataType>();

  data_profile profile = data_profile_builder().cardinality(0).no_validity().distribution(
    data_type_id, distribution_id::UNIFORM, 0, 100);
  auto keys_table = create_random_table({data_type_id}, row_count{num_rows}, profile);
  profile.set_null_probability(0.1);
  auto vals = create_random_column(data_type_id, row_count{num_rows}, profile);

  auto sort_order  = cudf::sorted_order(*keys_table);
  auto sorted_keys = cudf::gather(*keys_table, *sort_order);
  // No need to sort values using sort_order because they were generated randomly

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = vals->view();
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  state.add_global_memory_reads<nvbench::int8_t>(vals->alloc_size());
  std::size_t write_size = 0;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::groupby::groupby gb_obj(*sorted_keys, cudf::null_policy::EXCLUDE, cudf::sorted::YES);
    auto const result = gb_obj.aggregate(requests);
    write_size = result.first->alloc_size() + result.second.front().results.front()->alloc_size();
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  state.add_global_memory_writes<nvbench::int8_t>(write_size);
}

NVBENCH_BENCH_TYPES(bench_groupby_pre_sorted_sum, NVBENCH_TYPE_AXES(Types))
  .set_name("pre_sorted_sum")
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000});

static void bench_groupby_decimal128_sum_count_cardinality(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));

  data_profile const key_profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0, cardinality - 1);
  auto keys = create_random_column(cudf::type_to_id<int32_t>(), row_count{num_rows}, key_profile);
  // The random generator's cardinality option samples its dictionary with replacement. Seed one
  // copy of every key explicitly so the 127/128/129 axis measures the requested boundary exactly.
  auto const zero          = cudf::make_fixed_width_scalar<int32_t>(0);
  auto const one           = cudf::make_fixed_width_scalar<int32_t>(1);
  auto const distinct_keys = cudf::sequence(cardinality, *zero, *one);
  auto mutable_keys        = keys->mutable_view();
  cudf::copy_range_in_place(distinct_keys->view(), mutable_keys, 0, cardinality, 0);

  data_profile const value_profile =
    data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_to_id<numeric::decimal128>(),
      distribution_id::UNIFORM,
      0,
      1'000,
      numeric::scale_type{-4});
  auto const values = create_random_column(
    cudf::type_to_id<numeric::decimal128>(), row_count{num_rows}, value_profile);

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values->view();
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(
    cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));

  state.add_global_memory_reads<nvbench::int8_t>(keys->alloc_size() + values->alloc_size());
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::groupby::groupby gb(cudf::table_view{std::vector<cudf::column_view>{keys->view()}});
    auto const result = gb.aggregate(requests);
  });
  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / elapsed_time / 1'000'000., "Mrows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_groupby_decimal128_sum_count_cardinality)
  .set_name("decimal128_sum_count_cardinality")
  .add_int64_axis("num_rows", {30'600'000})
  .add_int64_axis("cardinality",
                  {64, 120, 127, 128, 129, 130, 160, 175, 256, 1'024, 4'096, 10'000, 1'000'000});
