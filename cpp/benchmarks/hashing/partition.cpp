/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace {

/** Seed used for the first independently generated key column. */
constexpr unsigned first_key_seed = 1;

/** Seed used for the first independently generated payload column. */
constexpr unsigned first_payload_seed = 1'000;

/** Integer key types exercised by the analytic benchmark. */
using analytic_key_types = nvbench::type_list<std::int32_t, std::int64_t>;

/**
 * @brief Create a deterministic random-data profile.
 *
 * A cardinality of zero generates values directly from each type's distribution. A positive
 * cardinality limits the number of distinct values.
 *
 * @param cardinality Maximum number of distinct values, or zero for no limit
 * @param null_probability Probability of a null, or no value for no validity mask
 * @return Data-generation profile
 */
data_profile make_profile(cudf::size_type cardinality,
                          std::optional<double> null_probability = std::nullopt)
{
  data_profile profile;
  profile.set_cardinality(cardinality);
  profile.set_avg_run_length(1);
  profile.set_null_probability(null_probability);
  return profile;
}

/**
 * @brief Append independently generated columns to an owning column vector.
 *
 * @param columns Destination column vector
 * @param types Types of columns to append
 * @param num_rows Number of rows in each column
 * @param profile Data-generation profile
 * @param seed Seed for the first column; incremented once per appended column
 */
void append_random_columns(std::vector<std::unique_ptr<cudf::column>>& columns,
                           std::vector<cudf::type_id> const& types,
                           cudf::size_type num_rows,
                           data_profile const& profile,
                           unsigned& seed)
{
  for (auto const type : types) {
    columns.push_back(create_random_column(type, row_count{num_rows}, profile, seed++));
  }
}

/**
 * @brief Create a table with independently generated key and payload columns.
 *
 * Key columns are generated first and have high cardinality by default. Payload columns are not
 * cardinality-limited. Separate seed ranges keep all columns independent and deterministic.
 *
 * @param key_types Types of the leading key columns
 * @param payload_types Types of the trailing payload columns
 * @param num_rows Number of table rows
 * @param key_null_probability Probability of a null in each key column
 * @param payload_null_probability Probability of a null in each payload column
 * @return Owning input table
 */
std::unique_ptr<cudf::table> make_input_table(
  std::vector<cudf::type_id> const& key_types,
  std::vector<cudf::type_id> const& payload_types,
  cudf::size_type num_rows,
  std::optional<double> key_null_probability     = std::nullopt,
  std::optional<double> payload_null_probability = std::nullopt)
{
  auto const key_profile     = make_profile(num_rows, key_null_probability);
  auto const payload_profile = make_profile(0, payload_null_probability);
  auto columns               = std::vector<std::unique_ptr<cudf::column>>{};
  columns.reserve(key_types.size() + payload_types.size());

  auto key_seed     = first_key_seed;
  auto payload_seed = first_payload_seed;
  append_random_columns(columns, key_types, num_rows, key_profile, key_seed);
  append_random_columns(columns, payload_types, num_rows, payload_profile, payload_seed);
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Return the indices of the leading key columns.
 *
 * @param num_keys Number of leading columns to select
 * @return Consecutive column indices beginning at zero
 */
std::vector<cudf::size_type> make_key_indices(cudf::size_type num_keys)
{
  auto keys = std::vector<cudf::size_type>(num_keys);
  std::iota(keys.begin(), keys.end(), cudf::size_type{0});
  return keys;
}

/**
 * @brief Run hash partitioning and register memory-traffic metrics.
 *
 * @param state NVBench state
 * @param input Owning input table
 * @param keys Indices of columns to hash
 * @param num_partitions Number of output partitions
 */
void run_hash_partition(nvbench::state& state,
                        std::unique_ptr<cudf::table> const& input,
                        std::vector<cudf::size_type> const& keys,
                        cudf::size_type num_partitions)
{
  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const input_bytes = static_cast<std::int64_t>(input->alloc_size());
  auto const key_bytes =
    std::accumulate(keys.begin(), keys.end(), std::int64_t{0}, [&](auto bytes, auto key) {
      return bytes + static_cast<std::int64_t>(input->get_column(key).alloc_size());
    });
  auto const output_bytes = input_bytes;
  auto const offset_bytes = static_cast<std::int64_t>(num_partitions + 1) * sizeof(cudf::size_type);

  state.add_element_count(input->num_rows(), "rows");
  state.add_buffer_size(input_bytes, "input_size", "input_size");
  state.add_buffer_size(key_bytes, "key_size", "key_size");
  state.add_buffer_size(output_bytes, "output_size", "output_size");
  state.add_buffer_size(offset_bytes, "offset_size", "offset_size");
  state.add_global_memory_reads<nvbench::int8_t>(input_bytes + key_bytes);
  state.add_global_memory_writes<nvbench::int8_t>(output_bytes + offset_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto output = cudf::hash_partition(input->view(), keys, num_partitions);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

/**
 * @brief Benchmark representative analytic tables with independently varied keys and payloads.
 *
 * @tparam Key Key element type
 * @param state NVBench state
 */
template <typename Key>
void bench_hash_partition_analytic(nvbench::state& state, nvbench::type_list<Key>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_keys = static_cast<cudf::size_type>(state.get_int64("num_key_columns"));
  auto const num_payload_columns =
    static_cast<cudf::size_type>(state.get_int64("num_payload_columns"));
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));

  auto input =
    make_input_table(std::vector<cudf::type_id>(num_keys, cudf::type_to_id<Key>()),
                     std::vector<cudf::type_id>(num_payload_columns, cudf::type_id::INT64),
                     num_rows);
  run_hash_partition(state, input, make_key_indices(num_keys), num_partitions);
}

/**
 * @brief Benchmark partition-count scaling through 4,096 partitions.
 *
 * @param state NVBench state
 */
void bench_hash_partition_partition_count(nvbench::state& state)
{
  auto const num_rows       = cudf::size_type{1} << 23;
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));
  auto input                = make_input_table(
    {cudf::type_id::INT64}, std::vector<cudf::type_id>(8, cudf::type_id::INT64), num_rows);
  run_hash_partition(state, input, {0}, num_partitions);
}

/**
 * @brief Benchmark column-count scaling at fixed input byte sizes.
 *
 * @param state NVBench state
 */
void bench_hash_partition_equal_input_size(nvbench::state& state)
{
  auto const target_input_bytes = state.get_int64("target_input_bytes");
  auto const num_columns        = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_partitions     = static_cast<cudf::size_type>(state.get_int64("num_partitions"));
  auto const row_width          = static_cast<std::int64_t>(num_columns) * sizeof(std::int64_t);

  if (target_input_bytes % row_width != 0) {
    state.skip("target_input_bytes must be divisible by the row width");
    return;
  }
  auto const num_rows = static_cast<cudf::size_type>(target_input_bytes / row_width);
  auto input          = make_input_table({cudf::type_id::INT64},
                                std::vector<cudf::type_id>(num_columns - 1, cudf::type_id::INT64),
                                num_rows);
  run_hash_partition(state, input, {0}, num_partitions);
}

/**
 * @brief Create a key column containing a requested fraction of one hot value.
 *
 * Non-hot rows contain an independently shuffled high-cardinality tail. The random Boolean mask
 * makes hot rows deterministic without forming long runs.
 *
 * @param num_rows Number of key rows
 * @param hot_probability Fraction of rows containing the hot value
 * @return Owning INT64 key column
 */
std::unique_ptr<cudf::column> make_hot_key_column(cudf::size_type num_rows, double hot_probability)
{
  auto const tail_profile = make_profile(num_rows);
  auto tail =
    create_random_column(cudf::type_id::INT64, row_count{num_rows}, tail_profile, first_key_seed);

  auto mask_profile = make_profile(0);
  mask_profile.set_bool_probability_true(hot_probability);
  auto mask = create_random_column(
    cudf::type_id::BOOL8, row_count{num_rows}, mask_profile, first_key_seed + 1);

  auto const hot_key = cudf::numeric_scalar<std::int64_t>{0};
  return cudf::copy_if_else(hot_key, tail->view(), mask->view());
}

/**
 * @brief Benchmark partition imbalance from low-cardinality and hot-key distributions.
 *
 * @param state NVBench state
 */
void bench_hash_partition_key_skew(nvbench::state& state)
{
  auto const num_rows       = cudf::size_type{1} << 23;
  auto const distribution   = state.get_string("distribution");
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));

  auto columns = std::vector<std::unique_ptr<cudf::column>>{};
  columns.reserve(9);
  if (distribution == "uniform_16_values") {
    auto const profile = make_profile(16);
    columns.push_back(
      create_random_column(cudf::type_id::INT64, row_count{num_rows}, profile, first_key_seed));
  } else if (distribution == "hot_50_percent") {
    columns.push_back(make_hot_key_column(num_rows, 0.5));
  } else if (distribution == "hot_90_percent") {
    columns.push_back(make_hot_key_column(num_rows, 0.9));
  } else {
    state.skip("Unknown key distribution");
    return;
  }

  auto const payload_profile = make_profile(0);
  auto payload_seed          = first_payload_seed;
  append_random_columns(columns,
                        std::vector<cudf::type_id>(8, cudf::type_id::INT64),
                        num_rows,
                        payload_profile,
                        payload_seed);
  auto input = std::make_unique<cudf::table>(std::move(columns));
  run_hash_partition(state, input, {0}, num_partitions);
}

/**
 * @brief Benchmark validity masks on keys, payload columns, and both.
 *
 * @param state NVBench state
 */
void bench_hash_partition_nullability(nvbench::state& state)
{
  auto const num_rows       = cudf::size_type{1} << 21;
  auto const nullable       = state.get_string("nullable");
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));
  constexpr double null_probability = 0.1;

  auto key_nulls     = std::optional<double>{};
  auto payload_nulls = std::optional<double>{};
  if (nullable == "keys") {
    key_nulls = null_probability;
  } else if (nullable == "payload") {
    payload_nulls = null_probability;
  } else if (nullable == "keys_and_payload") {
    key_nulls     = null_probability;
    payload_nulls = null_probability;
  } else {
    state.skip("Unknown nullability layout");
    return;
  }

  auto input = make_input_table({cudf::type_id::INT64},
                                std::vector<cudf::type_id>(8, cudf::type_id::INT64),
                                num_rows,
                                key_nulls,
                                payload_nulls);
  run_hash_partition(state, input, {0}, num_partitions);
}

/**
 * @brief Benchmark representative fixed-width, hybrid, and string-key tables.
 *
 * @param state NVBench state
 */
void bench_hash_partition_type_mix(nvbench::state& state)
{
  auto const num_rows       = cudf::size_type{1} << 21;
  auto const layout         = state.get_string("layout");
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));

  auto key_types     = std::vector<cudf::type_id>{};
  auto payload_types = std::vector<cudf::type_id>{};
  if (layout == "heterogeneous_fixed_payload") {
    key_types     = {cudf::type_id::INT64};
    payload_types = {cudf::type_id::INT8,
                     cudf::type_id::INT16,
                     cudf::type_id::INT32,
                     cudf::type_id::INT64,
                     cudf::type_id::FLOAT32,
                     cudf::type_id::FLOAT64,
                     cudf::type_id::DECIMAL128};
  } else if (layout == "hybrid_string_payload") {
    key_types     = {cudf::type_id::INT64};
    payload_types = {cudf::type_id::INT8,
                     cudf::type_id::INT16,
                     cudf::type_id::INT32,
                     cudf::type_id::INT64,
                     cudf::type_id::FLOAT32,
                     cudf::type_id::FLOAT64,
                     cudf::type_id::DECIMAL128,
                     cudf::type_id::STRING};
  } else if (layout == "string_key") {
    key_types     = {cudf::type_id::STRING};
    payload_types = std::vector<cudf::type_id>(8, cudf::type_id::INT64);
  } else if (layout == "mixed_composite_key") {
    key_types     = {cudf::type_id::INT64, cudf::type_id::STRING};
    payload_types = std::vector<cudf::type_id>(8, cudf::type_id::INT64);
  } else {
    state.skip("Unknown type layout");
    return;
  }

  auto input = make_input_table(key_types, payload_types, num_rows);
  run_hash_partition(
    state, input, make_key_indices(static_cast<cudf::size_type>(key_types.size())), num_partitions);
}

/**
 * @brief Benchmark DISTINCT-style partitioning that hashes every input column.
 *
 * @param state NVBench state
 */
void bench_hash_partition_all_keys_stress(nvbench::state& state)
{
  auto const num_rows       = cudf::size_type{1} << 21;
  auto const num_columns    = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));
  auto input =
    make_input_table(std::vector<cudf::type_id>(num_columns, cudf::type_id::INT64), {}, num_rows);
  run_hash_partition(state, input, make_key_indices(num_columns), num_partitions);
}

}  // namespace

NVBENCH_BENCH_TYPES(bench_hash_partition_analytic, NVBENCH_TYPE_AXES(analytic_key_types))
  .set_name("hash_partition_analytic")
  .set_type_axes_names({"key_type"})
  .add_int64_axis("num_rows", {1 << 17, 1 << 21, 1 << 24})
  .add_int64_axis("num_key_columns", {1, 2, 3})
  .add_int64_axis("num_payload_columns", {1, 8, 16})
  .add_int64_axis("num_partitions", {256, 1024});

NVBENCH_BENCH(bench_hash_partition_partition_count)
  .set_name("hash_partition_partition_count")
  .add_int64_axis("num_partitions", {64, 128, 256, 512, 1024, 2048, 4096});

NVBENCH_BENCH(bench_hash_partition_equal_input_size)
  .set_name("hash_partition_equal_input_size")
  .add_int64_axis("target_input_bytes",
                  {std::int64_t{256} << 20, std::int64_t{1} << 30, std::int64_t{4} << 30})
  .add_int64_axis("num_columns", {2, 8, 16, 256})
  .add_int64_axis("num_partitions", {256, 1024});

NVBENCH_BENCH(bench_hash_partition_key_skew)
  .set_name("hash_partition_key_skew")
  .add_string_axis("distribution", {"uniform_16_values", "hot_50_percent", "hot_90_percent"})
  .add_int64_axis("num_partitions", {256, 1024});

NVBENCH_BENCH(bench_hash_partition_nullability)
  .set_name("hash_partition_nullability")
  .add_string_axis("nullable", {"keys", "payload", "keys_and_payload"})
  .add_int64_axis("num_partitions", {256, 1024});

NVBENCH_BENCH(bench_hash_partition_type_mix)
  .set_name("hash_partition_type_mix")
  .add_string_axis(
    "layout",
    {"heterogeneous_fixed_payload", "hybrid_string_payload", "string_key", "mixed_composite_key"})
  .add_int64_axis("num_partitions", {256, 1024});

NVBENCH_BENCH(bench_hash_partition_all_keys_stress)
  .set_name("hash_partition_all_keys_stress")
  .add_int64_axis("num_columns", {2, 8, 16, 256})
  .add_int64_axis("num_partitions", {256, 1024});
