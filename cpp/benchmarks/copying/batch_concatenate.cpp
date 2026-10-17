/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

// ============================================================================
// Benchmark schema: a wide, deeply nested layout drawn by `make_schema_roots`.
// Roughly 1000 columns — flat leaves, single-level arrays, and random deep
// trees up to eight levels — generated from a fixed seed, so every run sees
// the same shape.
// ============================================================================

/// What a schema node produces: a leaf column, a list, a struct, or a map.
enum class node_kind { LEAF, ARRAY, STRUCT, MAP };

/// One node in the schema tree: a leaf column or a composite over `children`.
struct schema_node {
  node_kind kind{};
  cudf::type_id leaf_type{cudf::type_id::EMPTY};  ///< Only meaningful when kind == LEAF
  bool nullable{true};
  std::vector<schema_node> children;
};

/// Parameters controlling random data generation for a schema table.
struct table_gen_options {
  cudf::size_type num_rows{1024};
  double null_probability{0.1};        ///< Applied to nullable columns
  cudf::size_type max_list_length{5};  ///< Inclusive upper bound for array/map lengths
  cudf::size_type max_string_length{32};
  unsigned seed{1};
};

namespace {

/**
 * @brief Draw a leaf type with the benchmark's mix (mostly fixed-width, some strings).
 */
cudf::type_id draw_leaf_type(std::mt19937& rng)
{
  static std::array<cudf::type_id, 10> const leaves{cudf::type_id::INT64,
                                                    cudf::type_id::STRING,
                                                    cudf::type_id::FLOAT64,
                                                    cudf::type_id::BOOL8,
                                                    cudf::type_id::INT64,
                                                    cudf::type_id::FLOAT64,
                                                    cudf::type_id::STRING,
                                                    cudf::type_id::INT64,
                                                    cudf::type_id::FLOAT32,
                                                    cudf::type_id::TIMESTAMP_MICROSECONDS};
  return leaves[rng() % leaves.size()];
}

/**
 * @brief Recursively grow one subtree under a node budget.
 *
 * @param rng     Seeded generator (deterministic schema across runs)
 * @param depth   Remaining allowed depth below this node
 * @param budget  Nodes still available in the whole schema
 * @return The node's subtree, and the number of nodes consumed
 */
std::pair<schema_node, std::size_t> grow_subtree(std::mt19937& rng, int depth, std::size_t budget)
{
  schema_node node;
  if (depth <= 0 || budget <= 1 || (depth < 3 && (rng() % 8 == 0))) {
    node.leaf_type = draw_leaf_type(rng);
    return {std::move(node), 1};
  }
  auto const roll  = rng() % 10;
  std::size_t used = 1;
  if (roll < 4) {
    node.kind       = node_kind::ARRAY;
    auto [child, n] = grow_subtree(rng, depth - 1, budget - 1);
    used += n;
    node.children.push_back(std::move(child));
  } else if (roll < 6) {  // map: list<struct<key, value>>
    node.kind = node_kind::MAP;
    schema_node key;
    key.kind      = node_kind::LEAF;
    key.leaf_type = cudf::type_id::STRING;
    key.nullable  = false;
    schema_node val;
    val.kind      = node_kind::LEAF;
    val.leaf_type = draw_leaf_type(rng);
    val.nullable  = true;
    node.children.push_back(std::move(key));
    node.children.push_back(std::move(val));
    used += 2;
  } else {  // struct: the first child continues the spine at full depth
    node.kind      = node_kind::STRUCT;
    auto const fan = 1 + static_cast<int>(rng() % 3);
    for (int i = 0; i < fan && used < budget; ++i) {
      // The first child keeps the maximum depth (the spine); later ones stop early
      auto [child, n] =
        grow_subtree(rng, i == 0 ? depth - 1 : std::max(1, depth - 2), budget - used);
      used += n;
      node.children.push_back(std::move(child));
    }
  }
  return {std::move(node), used};
}

/**
 * @brief Build the benchmark's top-level schema: flat leaves, shallow arrays, deep trees.
 *
 * By default the schema spans a 1000-node budget (~115 top-level columns). Setting the
 * BATCH_CONCAT_BENCH_LARGE environment variable scales it to a ~8200-node budget
 * (~530 top-level columns, depth 8) — the magnitude of the production query shape this
 * benchmark stands in for, though not its actual schema.
 */
std::vector<schema_node> make_schema_roots(unsigned seed)
{
  std::mt19937 rng{seed};
  bool const large          = std::getenv("BATCH_CONCAT_BENCH_LARGE") != nullptr;
  std::size_t const kBudget = large ? 8200 : 1000;
  int const kFlatLeaves     = large ? 115 : 50;
  int const kShallowArrays  = large ? 40 : 9;

  std::vector<schema_node> roots;
  roots.reserve(large ? 560 : 120);
  std::size_t used = 0;

  for (int i = 0; i < kFlatLeaves; ++i) {
    schema_node node;
    node.leaf_type = draw_leaf_type(rng);
    roots.push_back(std::move(node));
    used += 1;
  }
  for (int i = 0; i < kShallowArrays; ++i) {
    schema_node arr;
    arr.kind = node_kind::ARRAY;
    schema_node elem;
    elem.leaf_type = draw_leaf_type(rng);
    arr.children.push_back(std::move(elem));
    roots.push_back(std::move(arr));
    used += 2;
  }
  // Deep trees until the budget is spent
  while (used + 2 <= kBudget) {
    auto [tree, n] = grow_subtree(rng, 8, kBudget - used);
    if (n < 2) { break; }
    roots.push_back(std::move(tree));
    used += n;
  }
  return roots;
}

}  // namespace

// ============================================================================
// Random data generation
// ============================================================================

data_profile make_leaf_profile(table_gen_options const& opts, bool nullable)
{
  auto builder = data_profile_builder()
                   .cardinality(0)
                   .distribution(cudf::type_id::INT64,
                                 distribution_id::NORMAL,
                                 static_cast<int64_t>(-1'000'000),
                                 static_cast<int64_t>(1'000'000))
                   .distribution(cudf::type_id::FLOAT64, distribution_id::NORMAL, -1.0e3, 1.0e3)
                   .distribution(cudf::type_id::FLOAT32, distribution_id::NORMAL, -1.0e3f, 1.0e3f)
                   .distribution(cudf::type_id::STRING,
                                 distribution_id::NORMAL,
                                 static_cast<uint32_t>(0),
                                 static_cast<uint32_t>(opts.max_string_length))
                   .distribution<cudf::timestamp_us>(cudf::type_id::TIMESTAMP_MICROSECONDS,
                                                     distribution_id::GEOMETRIC);
  if (nullable && opts.null_probability > 0.0) {
    builder.null_probability(opts.null_probability);
  } else {
    builder.no_validity();
  }
  return builder;
}

std::pair<cuda::device_buffer<cudf::bitmask_type>, cudf::size_type> make_null_mask(cudf::size_type num_rows,
                                                                          bool nullable,
                                                                          double null_probability,
                                                                          std::mt19937& rng)
{
  if (!nullable || null_probability <= 0.0 || num_rows == 0) {
    return {cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0};
  }
  std::bernoulli_distribution valid_dist(1.0 - null_probability);
  std::vector<bool> validity(static_cast<std::size_t>(num_rows));
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    validity[static_cast<std::size_t>(i)] = valid_dist(rng);
  }
  return cudf::test::detail::make_null_mask(validity.begin(), validity.end());
}

std::unique_ptr<cudf::column> create_column_from_schema(schema_node const& node,
                                                        table_gen_options const& opts,
                                                        unsigned seed);

std::unique_ptr<cudf::column> make_list_column(schema_node const& element_schema,
                                               table_gen_options const& opts,
                                               cudf::size_type num_rows,
                                               unsigned seed,
                                               bool nullable)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  std::mt19937 rng{seed};
  std::uniform_int_distribution<cudf::size_type> len_dist(0, opts.max_list_length);

  std::vector<cudf::size_type> offsets(static_cast<std::size_t>(num_rows) + 1, 0);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    offsets[static_cast<std::size_t>(i) + 1] = offsets[static_cast<std::size_t>(i)] + len_dist(rng);
  }
  auto const child_rows = offsets.back();

  auto child_opts     = opts;
  child_opts.num_rows = child_rows;
  auto child          = create_column_from_schema(element_schema, child_opts, seed + 1);

  auto d_offsets = cudf::detail::make_device_uvector(offsets, stream, mr);
  auto offsets_column =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   num_rows + 1,
                                   d_offsets.release(),
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                   0);

  auto [null_mask, null_count] = make_null_mask(num_rows, nullable, opts.null_probability, rng);

  return cudf::make_lists_column(
    num_rows, std::move(offsets_column), std::move(child), null_count, std::move(null_mask));
}

std::unique_ptr<cudf::column> make_struct_column(std::vector<schema_node> const& fields,
                                                 table_gen_options const& opts,
                                                 cudf::size_type num_rows,
                                                 unsigned seed,
                                                 bool nullable)
{
  CUDF_EXPECTS(!fields.empty(), "Struct schema must have at least one child field");
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  std::mt19937 rng{seed};

  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(fields.size());
  auto field_opts     = opts;
  field_opts.num_rows = num_rows;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    children.push_back(
      create_column_from_schema(fields[i], field_opts, seed + static_cast<unsigned>(i) + 1));
  }

  auto [null_mask, null_count] = make_null_mask(num_rows, nullable, opts.null_probability, rng);

  return cudf::make_structs_column(
    num_rows, std::move(children), null_count, std::move(null_mask), stream, mr);
}

std::unique_ptr<cudf::column> create_column_from_schema(schema_node const& node,
                                                        table_gen_options const& opts,
                                                        unsigned seed)
{
  auto const num_rows = opts.num_rows;
  switch (node.kind) {
    case node_kind::LEAF: {
      auto const profile = make_leaf_profile(opts, node.nullable);
      return create_random_column(node.leaf_type, row_count{num_rows}, profile, seed);
    }
    case node_kind::STRUCT:
      return make_struct_column(node.children, opts, num_rows, seed, node.nullable);
    case node_kind::ARRAY: {
      CUDF_EXPECTS(node.children.size() == 1,
                   "Array schema must have exactly one element child, got " +
                     std::to_string(node.children.size()));
      return make_list_column(node.children.front(), opts, num_rows, seed, node.nullable);
    }
    case node_kind::MAP: {
      CUDF_EXPECTS(
        node.children.size() == 2,
        "Map schema must have key and value children, got " + std::to_string(node.children.size()));
      schema_node entry;
      entry.kind     = node_kind::STRUCT;
      entry.nullable = false;
      entry.children = node.children;
      return make_list_column(entry, opts, num_rows, seed, node.nullable);
    }
  }
  CUDF_FAIL("Unhandled node kind");
}

std::unique_ptr<cudf::table> create_schema_table(table_gen_options const& opts)
{
  auto schema = make_schema_roots(opts.seed);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(schema.size());
  for (std::size_t i = 0; i < schema.size(); ++i) {
    columns.push_back(
      create_column_from_schema(schema[i], opts, opts.seed + static_cast<unsigned>(i) * 97u));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

// ============================================================================
// Benchmark
// ============================================================================

/**
 * @brief Concatenates the two inputs. Which implementation runs is fixed at
 * compile time: `cudf::batch_concatenate` when `BATCH_CONCAT_BENCH_IMPL_BATCH`
 * is defined, `cudf::concatenate` otherwise.
 */
std::unique_ptr<cudf::table> dispatch_concatenate(std::vector<cudf::table_view> const& views,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
#if defined(BATCH_CONCAT_BENCH_IMPL_BATCH)
  return cudf::batch_concatenate(views, stream, mr);
#else
  return cudf::concatenate(views, stream, mr);
#endif
}

void bench_wide_nested_concatenate(nvbench::state& state)
{
  table_gen_options opts;
  opts.num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  opts.null_probability = state.get_float64("nulls");
  opts.max_list_length  = static_cast<cudf::size_type>(state.get_int64("max_list_length"));
  opts.seed             = 42;

  auto table = create_schema_table(opts);
  std::vector<cudf::table_view> tables{table->view(), table->view()};

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  // Rough traffic estimate: rows * columns * 8 bytes * 2 inputs
  auto const nbytes = static_cast<int64_t>(opts.num_rows) * table->num_columns() * 8 * 2;
  state.add_global_memory_reads<int8_t>(nbytes);
  state.add_global_memory_writes<int8_t>(nbytes);

  auto const make_result = [&](rmm::device_async_resource_ref mr) {
    return dispatch_concatenate(tables, stream, mr);
  };
  auto const concatenate_once = [&](rmm::device_async_resource_ref mr) {
    [[maybe_unused]] auto result = make_result(mr);
  };

  // Peak memory from one untimed call: keeping the statistics adaptor installed through the
  // timed loop would tax every allocation and free, thousands per call on this schema.
  {
    auto const mem_stats_logger = cudf::memory_stats_logger();
    concatenate_once(cudf::get_current_device_resource_ref());
    cudf::detail::sync_stream(stream);
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  }

  auto const mr = cudf::get_current_device_resource_ref();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) { concatenate_once(mr); });
}

}  // namespace

NVBENCH_BENCH(bench_wide_nested_concatenate)
  .set_name("wide_nested_concatenate")
  .add_int64_axis("num_rows", {256, 1024, 4096, 16'384, 65'536, 100'000})
  .add_int64_axis("max_list_length", {3})
  .add_float64_axis("nulls", {0.1});

// ============================================================================
// Delegation probe: measures the small-payload delegation decision.
//
// A flat fixed-width input whose payload is below `min_batch_payload_bytes`
// (8 MiB) delegates to the standard concatenate; above it, the batched
// engine runs. This probe sweeps the payload around that threshold with the
// unmodified engine, so the step in the curve is the delegation itself.
// ============================================================================

void bench_flat_delegation(nvbench::state& state)
{
  auto const rows_per_col = static_cast<cudf::size_type>(state.get_int64("rows_per_col"));
  auto const num_cols     = static_cast<cudf::size_type>(state.get_int64("num_cols"));

  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> wrappers;
  wrappers.reserve(num_cols);
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    // Distinct values per column so nothing is optimized away
    std::vector<int32_t> vals(rows_per_col);
    std::iota(vals.begin(), vals.end(), i * rows_per_col);
    wrappers.emplace_back(vals.begin(), vals.end());
  }
  std::vector<cudf::column_view> columns;
  columns.reserve(num_cols);
  for (auto const& w : wrappers) {
    columns.push_back(w);
  }

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  auto const nbytes = static_cast<int64_t>(rows_per_col) * num_cols * 4 * 2;
  state.add_global_memory_reads<int8_t>(nbytes);
  state.add_global_memory_writes<int8_t>(nbytes);

  auto const make_result = [&](rmm::device_async_resource_ref mr) {
#if defined(BATCH_CONCAT_BENCH_IMPL_BATCH)
    return cudf::batch_concatenate(columns, stream, mr);
#else
    return cudf::concatenate(columns, stream, mr);
#endif
  };
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto r = make_result(cudf::get_current_device_resource_ref());
  });
}

NVBENCH_BENCH(bench_flat_delegation)
  .set_name("flat_delegation")
  .add_int64_axis("rows_per_col", {1024, 65536, 262144, 1048576})  // 4KB, 256KB, 1MB, 4MB per col
  .add_int64_axis("num_cols", {8, 32})
  .add_int64_axis("max_list_length", {3})
  .add_float64_axis("nulls", {0.0});
