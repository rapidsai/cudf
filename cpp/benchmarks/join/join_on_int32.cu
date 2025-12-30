/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/table_utilities.hpp>
#include <benchmarks/join/join_common.hpp>

#include <cudf/join/join.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @brief Benchmark for join performance on INT32 keys.
 *
 * This simulates the join phase after key remapping, where complex keys have
 * been converted to simple INT32 IDs. Comparing this with key_remap_build times
 * shows what percentage of total join time is spent on metrics computation.
 */

// Cardinality distributions (same as key_remap_build)
enum class Cardinality {
  ALL_UNIQUE,
  HIGH_UNIQUE,
  MED_UNIQUE,
  LOW_UNIQUE,
  SINGLE_KEY
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

enum class JoinAlgo { HASH, SORT_MERGE };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  JoinAlgo,
  [](auto value) {
    switch (value) {
      case JoinAlgo::HASH: return "HASH";
      case JoinAlgo::SORT_MERGE: return "SORT_MERGE";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

template <Cardinality Card, JoinAlgo Algo>
void nvbench_join_on_int32(nvbench::state& state,
                           nvbench::type_list<nvbench::enum_type<Card>,
                                              nvbench::enum_type<Algo>>)
{
  auto const num_rows = state.get_int64("num_rows");

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

  // Generate INT32 key tables (simulating remapped keys)
  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(data_type::INT32)), 1);

  auto [left_table, right_table] = generate_input_tables<false>(
    dtypes, num_rows, num_rows, 0, multiplicity, selectivity);

  auto const left  = left_table->view();
  auto const right = right_table->view();

  auto const input_size = estimate_size(left) + estimate_size(right);
  state.add_element_count(input_size, "input_size");
  state.add_global_memory_reads<nvbench::int8_t>(input_size);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  if constexpr (Algo == JoinAlgo::HASH) {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = cudf::inner_join(left, right, cudf::null_equality::EQUAL);
    });
  } else {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto smj    = cudf::sort_merge_join(right, cudf::sorted::NO, cudf::null_equality::EQUAL);
      auto result = smj.inner_join(left, cudf::sorted::NO);
    });
  }
}

// Cardinality distributions
using cardinality_list = nvbench::enum_type_list<Cardinality::ALL_UNIQUE,
                                                 Cardinality::HIGH_UNIQUE,
                                                 Cardinality::MED_UNIQUE,
                                                 Cardinality::LOW_UNIQUE,
                                                 Cardinality::SINGLE_KEY>;

// Join algorithms
using join_algo_list = nvbench::enum_type_list<JoinAlgo::HASH, JoinAlgo::SORT_MERGE>;

NVBENCH_BENCH_TYPES(nvbench_join_on_int32,
                    NVBENCH_TYPE_AXES(cardinality_list, join_algo_list))
  .set_name("join_on_int32")
  .set_type_axes_names({"Cardinality", "JoinAlgo"})
  .add_int64_axis("num_rows", {1'000'000, 10'000'000, 100'000'000});

