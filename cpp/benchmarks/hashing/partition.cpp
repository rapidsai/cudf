/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

enum class key_selection { first, all };

static void run_hash_partition(nvbench::state& state,
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

  state.add_buffer_size(input_bytes, "input_size", "input_size");
  state.add_buffer_size(key_bytes, "key_size", "key_size");
  state.add_buffer_size(output_bytes, "output_size", "output_size");
  state.add_buffer_size(offset_bytes, "offset_size", "offset_size");
  state.add_global_memory_reads<nvbench::int8_t>(input_bytes + key_bytes);
  state.add_global_memory_writes<nvbench::int8_t>(output_bytes + offset_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto output = cudf::hash_partition(input->view(), keys, num_partitions);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

static void bench_hash_partition_impl(nvbench::state& state, key_selection selection)
{
  using T = int64_t;

  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols         = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const num_partitions   = static_cast<cudf::size_type>(state.get_int64("num_partitions"));
  auto const null_probability = state.get_float64("null_probability");

  // Create owning columns
  auto input_table = create_sequence_table(
    cycle_dtypes({cudf::type_to_id<T>()}, num_cols),
    row_count{static_cast<cudf::size_type>(num_rows)},
    null_probability == 0.0 ? std::nullopt : std::optional<double>{null_probability});

  auto const num_keys  = selection == key_selection::all ? num_cols : cudf::size_type{1};
  auto columns_to_hash = std::vector<cudf::size_type>(num_keys);
  std::iota(columns_to_hash.begin(), columns_to_hash.end(), 0);
  run_hash_partition(state, input_table, columns_to_hash, num_partitions);
}

static void bench_hash_partition_all_keys(nvbench::state& state)
{
  bench_hash_partition_impl(state, key_selection::all);
}

static void bench_hash_partition_single_key(nvbench::state& state)
{
  bench_hash_partition_impl(state, key_selection::first);
}

static void bench_mixed_payload(nvbench::state& state)
{
  auto input =
    create_random_table({cudf::type_id::INT32,
                         cudf::type_id::INT8,
                         cudf::type_id::INT16,
                         cudf::type_id::INT32,
                         cudf::type_id::INT64,
                         cudf::type_id::FLOAT32,
                         cudf::type_id::FLOAT64,
                         cudf::type_id::DECIMAL128,
                         cudf::type_id::STRING},
                        row_count{1 << 21},
                        data_profile_builder().cardinality(0).avg_run_length(1).no_validity());
  run_hash_partition(state, input, {0}, 1024);
}

NVBENCH_BENCH(bench_hash_partition_all_keys)
  .set_name("hash_partition")
  .add_int64_axis("num_rows", {1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21})
  .add_int64_axis("num_cols", {1, 4, 8, 16, 256})
  .add_int64_axis("num_partitions", {64, 128, 256, 512, 1024})
  .add_float64_axis("null_probability", {0.0});

NVBENCH_BENCH(bench_hash_partition_single_key)
  .set_name("hash_partition_narrow_table_large_row_count")
  .add_int64_axis("num_rows", {1 << 26})
  .add_int64_axis("num_cols", {2})
  .add_int64_axis("num_partitions", {1024})
  .add_float64_axis("null_probability", {0.0});

NVBENCH_BENCH(bench_hash_partition_single_key)
  .set_name("hash_partition_shmoo_partition_count")
  .add_int64_axis("num_rows", {1 << 21})
  .add_int64_axis("num_cols", {8})
  .add_int64_axis("num_partitions", {64, 256, 1024, 2048})
  .add_float64_axis("null_probability", {0.0});

NVBENCH_BENCH(bench_hash_partition_single_key)
  .set_name("hash_partition_nullable_fixed_width")
  .add_int64_axis("num_rows", {1 << 21})
  .add_int64_axis("num_cols", {8})
  .add_int64_axis("num_partitions", {1024})
  .add_float64_axis("null_probability", {0.1});

NVBENCH_BENCH(bench_mixed_payload).set_name("hash_partition_mixed_payload");
