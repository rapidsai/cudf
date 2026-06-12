/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "variant_blob_builders.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <string>
#include <vector>

namespace {

using cudf::io::parquet::benchmark_util::build_bare_int32_value;
using cudf::io::parquet::benchmark_util::build_metadata;
using cudf::io::parquet::benchmark_util::build_mixed_object_value;
using cudf::io::parquet::benchmark_util::build_nested_object_value;
using cudf::io::parquet::benchmark_util::build_object_value;
using cudf::io::parquet::benchmark_util::build_variant_column;

// ---------------------------------------------------------------------------
// Key naming with 5 distinct length groups
// ---------------------------------------------------------------------------

// Dictionary sizes must be divisible by NUM_GROUPS.
constexpr int NUM_GROUPS              = 5;
constexpr int KEY_LENGTHS[NUM_GROUPS] = {3, 6, 10, 15, 21};

// Produce a key name for global index `idx` within a dictionary of `dict_size`
// keys.  Keys are split into NUM_GROUPS equal blocks; each block uses a
// different target length so the dictionary scan's length-mismatch early-exit
// is exercised realistically.
//
//   Group 0 (len  3): "a0_", "a1_", ..., "a<gs-1>"
//   Group 1 (len  6): "b0____", ...
//   Group 2 (len 10): "c0________", ...
//   Group 3 (len 15): "d0_____________", ...
//   Group 4 (len 21): "e0___________________", ...
//
std::string key_name(int idx, int dict_size)
{
  int const group_size = dict_size / NUM_GROUPS;
  int const group      = idx / group_size;
  int const pos        = idx % group_size;
  int const target_len = KEY_LENGTHS[group];

  std::string s = std::string(1, static_cast<char>('a' + group)) + std::to_string(pos);
  if (static_cast<int>(s.size()) < target_len) {
    s.append(static_cast<std::size_t>(target_len) - s.size(), '_');
  }
  return s;
}

// All key names for a full dictionary.
std::vector<std::string> full_keys(int dict_size)
{
  std::vector<std::string> out;
  out.reserve(dict_size);
  for (int i = 0; i < dict_size; ++i) {
    out.push_back(key_name(i, dict_size));
  }
  return out;
}

// Dictionary indices of the 5-key subset within the full dictionary.
// Picks one key from each length group, always including index 0 (first)
// and index full_size-1 (last) so that both "first key" and "last key"
// benchmarks find the target in the subset.
std::vector<int> subset_field_ids(int full_size)
{
  int const gs = full_size / NUM_GROUPS;
  return {0, gs + gs / 2, 2 * gs + gs / 2, 3 * gs + gs / 2, full_size - 1};
}

// 5-key subset as key-name strings.
std::vector<std::string> subset_keys(int full_size)
{
  auto const ids = subset_field_ids(full_size);
  std::vector<std::string> out;
  out.reserve(ids.size());
  for (int id : ids) {
    out.push_back(key_name(id, full_size));
  }
  return out;
}

// Sequential field IDs 0..n-1.
std::vector<int> seq_ids(int n)
{
  std::vector<int> out(n);
  for (int i = 0; i < n; ++i) {
    out[i] = i;
  }
  return out;
}

// Variant binary blob builders and build_variant_column now live in
// variant_blob_builders.hpp; only host-side scenario helpers remain here.

}  // namespace

// ---------------------------------------------------------------------------
// Benchmark: get_variant_field with divergence scenarios
// ---------------------------------------------------------------------------
//
// Each scenario targets the LAST key in its dictionary, forcing a full linear
// scan.  The 5-key "subset" dictionary is always drawn from the same key space
// as the larger dictionary (last key of each length group), so the same target
// key string resolves in both.
//
// Scenarios are designed to isolate one divergence axis each:
//
//   uniform_small       5-key subset, 5 fields       — small-work baseline
//   uniform_large       50-key full,  50 fields      — large-work baseline (no divergence)
//   skewed_field_count  50-key full for ALL rows;    — pure field-ID scan divergence
//                       even=5 fields, odd=50 fields
//   skewed_dict_size    even=5-key/5 fields,         — pure dict-scan divergence
//                       odd=100-key/5 fields
//   half_missing        20-key full for ALL rows;    — pure found-vs-null divergence
//                       even=target present, odd=absent
//

static void bench_get_variant_field(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const scenario = state.get_string("scenario");
  auto const key_pos  = state.get_string("key_position");
  bool const first    = (key_pos == "first");

  std::vector<std::vector<uint8_t>> meta_rows(num_rows);
  std::vector<std::vector<uint8_t>> val_rows(num_rows);
  std::string target_key;

  // Helper: pick first or last key from a key list.
  auto pick_target = [&](std::vector<std::string> const& keys) {
    return first ? keys.front() : keys.back();
  };

  if (scenario == "uniform_small") {
    // 5-key subset of the 50-key space.  5 fields.
    auto const keys      = subset_keys(50);
    auto const fids      = seq_ids(5);
    target_key           = pick_target(keys);
    auto const meta_blob = build_metadata(keys);
    auto const val_blob  = build_object_value(fids);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      meta_rows[i] = meta_blob;
      val_rows[i]  = val_blob;
    }
  } else if (scenario == "uniform_large") {
    // Full 50-key dict.  50 fields.
    auto const keys      = full_keys(50);
    auto const fids      = seq_ids(50);
    target_key           = pick_target(keys);
    auto const meta_blob = build_metadata(keys);
    auto const val_blob  = build_object_value(fids);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      meta_rows[i] = meta_blob;
      val_rows[i]  = val_blob;
    }
  } else if (scenario == "skewed_field_count") {
    // SAME 50-key dict for every row — dict scan cost is identical.
    // Even rows: 5 fields (subset positions).  Odd rows: 50 fields.
    // Isolates field-ID scan divergence.
    auto const keys      = full_keys(50);
    target_key           = pick_target(keys);
    auto const meta_blob = build_metadata(keys);
    auto const sub_fids  = subset_field_ids(50);
    auto const all_fids  = seq_ids(50);
    auto const val_small = build_object_value(sub_fids);
    auto const val_large = build_object_value(all_fids);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      meta_rows[i] = meta_blob;
      val_rows[i]  = (i % 2 == 0) ? val_small : val_large;
    }
  } else if (scenario == "skewed_dict_size") {
    // Even: 5-key subset dict, 5 fields.  Odd: 100-key full dict, 5 fields.
    // Field count is 5 for both — isolates dict-scan divergence.
    // The subset includes both the first and last key of the full dict,
    // so the target is present in both row types.
    auto const keys_sub  = subset_keys(100);
    auto const keys_full = full_keys(100);
    target_key           = pick_target(keys_sub);
    auto const meta_sub  = build_metadata(keys_sub);
    auto const meta_full = build_metadata(keys_full);
    auto const fids_sub  = seq_ids(5);
    auto const fids_full = subset_field_ids(100);
    auto const val_sub   = build_object_value(fids_sub);
    auto const val_full  = build_object_value(fids_full);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      if (i % 2 == 0) {
        meta_rows[i] = meta_sub;
        val_rows[i]  = val_sub;
      } else {
        meta_rows[i] = meta_full;
        val_rows[i]  = val_full;
      }
    }
  } else if (scenario == "half_missing") {
    // SAME 20-key dict for every row — dict scan cost is identical.
    // Even: all 20 fields including target.  Odd: 19 fields, target removed.
    // Isolates found-vs-null divergence.
    auto const keys          = full_keys(20);
    int const target_fid     = first ? 0 : static_cast<int>(keys.size()) - 1;
    target_key               = keys[target_fid];
    auto const meta_blob     = build_metadata(keys);
    auto const all_fids      = seq_ids(20);
    auto const val_present   = build_object_value(all_fids);
    auto const val_no_target = build_object_value(all_fids, target_fid);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      meta_rows[i] = meta_blob;
      val_rows[i]  = (i % 2 == 0) ? val_present : val_no_target;
    }
  }

  auto col = build_variant_column(meta_rows, val_rows, stream, mr);
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result =
      cudf::io::parquet::experimental::get_variant_field(col->view(), target_key, stream, mr);
  });
}

NVBENCH_BENCH(bench_get_variant_field)
  .set_name("get_variant_field")
  .add_int64_power_of_two_axis("num_rows", {15, 17, 19, 21})
  .add_string_axis(
    "scenario",
    {"uniform_small", "uniform_large", "skewed_field_count", "skewed_dict_size", "half_missing"})
  .add_string_axis("key_position", {"first", "last"});

// ---------------------------------------------------------------------------
// Benchmark: cast_variant (bare INT32 decode throughput baseline)
// ---------------------------------------------------------------------------

static void bench_cast_variant(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  auto const meta_blob = build_metadata({});
  auto const val_blob  = build_bare_int32_value(42);

  std::vector<std::vector<uint8_t>> meta_rows(num_rows, meta_blob);
  std::vector<std::vector<uint8_t>> val_rows(num_rows, val_blob);

  auto col = build_variant_column(meta_rows, val_rows, stream, mr);
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::io::parquet::experimental::cast_variant(
      col->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
  });
}

NVBENCH_BENCH(bench_cast_variant)
  .set_name("cast_variant_int32")
  .add_int64_power_of_two_axis("num_rows", {15, 17, 19, 21});

// ---------------------------------------------------------------------------
// Benchmark: get_variant_field with a nested (multi-key) path
// ---------------------------------------------------------------------------
//
// Measures how the fused path-walking kernel amortizes across increasing path
// depth.  Dictionary and per-level field count match the `uniform_large/first`
// case of `bench_get_variant_field` (50 keys, 50 fields per object, target at
// field id 0 in every level), so depth=1 is directly comparable to that cell.
//

static void bench_get_variant_field_nested(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const depth    = static_cast<int>(state.get_int64("depth"));
  auto const key_pos  = state.get_string("key_position");
  bool const first    = (key_pos == "first");

  constexpr int n_fields = 50;
  int const target_fid   = first ? 0 : (n_fields - 1);
  auto const keys        = full_keys(n_fields);
  auto const meta_blob   = build_metadata(keys);
  auto const val_blob    = build_nested_object_value(n_fields, depth, target_fid);

  std::vector<std::vector<uint8_t>> meta_rows(num_rows, meta_blob);
  std::vector<std::vector<uint8_t>> val_rows(num_rows, val_blob);

  // Build a JSONPath-like "$.key.key. ... .key" with the target key repeated `depth` times.
  std::string path = "$";
  for (int i = 0; i < depth; ++i) {
    path.push_back('.');
    path.append(keys[target_fid]);
  }

  auto col = build_variant_column(meta_rows, val_rows, stream, mr);
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::io::parquet::experimental::get_variant_field(col->view(), path, stream, mr);
  });
}

NVBENCH_BENCH(bench_get_variant_field_nested)
  .set_name("get_variant_field_nested")
  .add_int64_power_of_two_axis("num_rows", {17, 19, 21})
  .add_int64_axis("depth", {1, 2, 3, 4})
  .add_string_axis("key_position", {"first", "last"});
