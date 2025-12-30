/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/table_utilities.hpp>
#include <benchmarks/join/join_common.hpp>

#include <cudf/join/key_remapping.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @brief Benchmark for key_remapping build phase with metrics computation.
 *
 * This benchmark isolates the key_remapping construction time (including metrics)
 * to compare different metrics computation algorithms across various data distributions.
 */

// Cardinality distributions for testing
enum class Cardinality {
  ALL_UNIQUE,   // 100% unique keys (multiplicity = 1)
  HIGH_UNIQUE,  // 10% unique keys (multiplicity = 10)
  MED_UNIQUE,   // 1% unique keys (multiplicity = 100)
  LOW_UNIQUE,   // 0.1% unique keys (multiplicity = 1000)
  SINGLE_KEY    // 1 unique key (all duplicates)
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  Cardinality,
  [](auto value) {
    switch (value) {
      case Cardinality::ALL_UNIQUE: return "ALL_UNIQUE";
      case Cardinality::HIGH_UNIQUE: return "HIGH_UNIQUE";
      case Cardinality::MED_UNIQUE: return "MED_UNIQUE";
      case Cardinality::LOW_UNIQUE: return "LOW_UNIQUE";
      case Cardinality::SINGLE_KEY: return "SINGLE_KEY";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::key_remap_metrics_algo,
  [](auto value) {
    switch (value) {
      case cudf::key_remap_metrics_algo::SORT_REDUCE: return "SORT_REDUCE";
      case cudf::key_remap_metrics_algo::ATOMIC: return "ATOMIC";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

template <bool Nullable, data_type DataType, Cardinality Card, cudf::key_remap_metrics_algo Algo>
void nvbench_key_remap_build(nvbench::state& state,
                             nvbench::type_list<nvbench::enum_type<Nullable>,
                                                nvbench::enum_type<DataType>,
                                                nvbench::enum_type<Card>,
                                                nvbench::enum_type<Algo>>)
{
  auto const num_rows = state.get_int64("num_rows");
  auto const num_keys = state.get_int64("num_keys");
  auto dtypes         = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  // Determine multiplicity based on cardinality
  int multiplicity = 1;
  if constexpr (Card == Cardinality::ALL_UNIQUE) {
    multiplicity = 1;
  } else if constexpr (Card == Cardinality::HIGH_UNIQUE) {
    multiplicity = 10;
  } else if constexpr (Card == Cardinality::MED_UNIQUE) {
    multiplicity = 100;
  } else if constexpr (Card == Cardinality::LOW_UNIQUE) {
    multiplicity = 1000;
  } else if constexpr (Card == Cardinality::SINGLE_KEY) {
    multiplicity = num_rows;
  }

  double selectivity = 0.5;

  // Generate table
  auto [table, _] = generate_input_tables<Nullable>(
    dtypes, num_rows, num_rows, 0, multiplicity, selectivity);
  auto const keys = table->view();

  auto const input_size = estimate_size(keys);
  state.add_element_count(input_size, "input_size");
  state.add_global_memory_reads<nvbench::int8_t>(input_size);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::key_remapping remap(keys,
                              cudf::null_equality::EQUAL,
                              true,  // compute_metrics
                              Algo,
                              cudf::get_default_stream());
    // Access metrics to ensure they're computed
    auto dc = remap.get_distinct_count();
    auto mc = remap.get_max_duplicate_count();
    (void)dc;
    (void)mc;
  });
}

// Data types to test
using key_remap_datatypes = nvbench::enum_type_list<data_type::INT32,
                                                    data_type::INT64,
                                                    data_type::STRING,
                                                    data_type::STRUCT>;

// Cardinality distributions
using cardinality_list = nvbench::enum_type_list<Cardinality::ALL_UNIQUE,
                                                 Cardinality::HIGH_UNIQUE,
                                                 Cardinality::MED_UNIQUE,
                                                 Cardinality::LOW_UNIQUE,
                                                 Cardinality::SINGLE_KEY>;

// Metrics algorithms
using algo_list = nvbench::enum_type_list<cudf::key_remap_metrics_algo::SORT_REDUCE,
                                          cudf::key_remap_metrics_algo::ATOMIC>;

// Nullable options
using nullable_list = nvbench::enum_type_list<false>;

NVBENCH_BENCH_TYPES(nvbench_key_remap_build,
                    NVBENCH_TYPE_AXES(nullable_list,
                                      key_remap_datatypes,
                                      cardinality_list,
                                      algo_list))
  .set_name("key_remap_build")
  .set_type_axes_names({"Nullable", "DataType", "Cardinality", "Algorithm"})
  .add_int64_axis("num_rows", {1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("num_keys", {1, 2, 3});

