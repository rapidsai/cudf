/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32_jit_tags.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <array>
#include <memory>
#include <unordered_set>

namespace cudf::hashing::detail {

namespace {

inline cudf::detail::jit_lto::LauncherJitCache& murmur_jit_launcher_cache()
{
  static cudf::detail::jit_lto::LauncherJitCache cache;
  return cache;
}

// Every `type_id` handled by `murmur_jit_hash_dispatcher` must have exactly one device
// definition of `murmur_jit_hasher<T>` in the nvJitLink input. Types present in the table use the
// real hasher fatbin; all others use a separate strong no-op fatbin (nvJitLink does not merge
// weak + strong overrides for the same symbol).
static constexpr std::array<type_id, 30> murmur_jit_hasher_type_ids{{
  type_id::INT8,
  type_id::INT16,
  type_id::INT32,
  type_id::INT64,
  type_id::UINT8,
  type_id::UINT16,
  type_id::UINT32,
  type_id::UINT64,
  type_id::FLOAT32,
  type_id::FLOAT64,
  type_id::BOOL8,
  type_id::TIMESTAMP_DAYS,
  type_id::TIMESTAMP_SECONDS,
  type_id::TIMESTAMP_MILLISECONDS,
  type_id::TIMESTAMP_MICROSECONDS,
  type_id::TIMESTAMP_NANOSECONDS,
  type_id::DURATION_DAYS,
  type_id::DURATION_SECONDS,
  type_id::DURATION_MILLISECONDS,
  type_id::DURATION_MICROSECONDS,
  type_id::DURATION_NANOSECONDS,
  type_id::DICTIONARY32,
  type_id::STRING,
  type_id::LIST,
  type_id::DECIMAL32,
  type_id::DECIMAL64,
  type_id::DECIMAL128,
  type_id::STRUCT,
}};

inline void insert_logical_type(std::unordered_set<type_id>& ids, column_view const& col)
{
  ids.insert(col.type().id());
  if (col.type().id() == type_id::DICTIONARY32) {
    dictionary_column_view const dcv(col);
    insert_logical_type(ids, dcv.keys());
  }
}

inline void collect_table_logical_types(std::unordered_set<type_id>& ids, table_view const& table)
{
  for (size_type c = 0; c < table.num_columns(); ++c) {
    insert_logical_type(ids, table.column(c));
  }
}

inline void add_strong_hasher_fragment(cudf::detail::jit_lto::AlgorithmPlanner& planner, type_id id)
{
  using namespace cudf::hashing::detail::jit_lto;
  switch (id) {
    case type_id::INT8: planner.add_static_fragment<fragment_tag_murmur_hasher<tag_i8>>(); break;
    case type_id::INT16: planner.add_static_fragment<fragment_tag_murmur_hasher<tag_i16>>(); break;
    case type_id::INT32: planner.add_static_fragment<fragment_tag_murmur_hasher<tag_i32>>(); break;
    case type_id::INT64: planner.add_static_fragment<fragment_tag_murmur_hasher<tag_i64>>(); break;
    case type_id::UINT8: planner.add_static_fragment<fragment_tag_murmur_hasher<tag_u8>>(); break;
    case type_id::UINT16: planner.add_static_fragment<fragment_tag_murmur_hasher<tag_u16>>(); break;
    case type_id::UINT32: planner.add_static_fragment<fragment_tag_murmur_hasher<tag_u32>>(); break;
    case type_id::UINT64: planner.add_static_fragment<fragment_tag_murmur_hasher<tag_u64>>(); break;
    case type_id::FLOAT32:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_f32>>();
      break;
    case type_id::FLOAT64:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_f64>>();
      break;
    case type_id::BOOL8: planner.add_static_fragment<fragment_tag_murmur_hasher<tag_b8>>(); break;
    case type_id::TIMESTAMP_DAYS:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_ts_day>>();
      break;
    case type_id::TIMESTAMP_SECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_ts_s>>();
      break;
    case type_id::TIMESTAMP_MILLISECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_ts_ms>>();
      break;
    case type_id::TIMESTAMP_MICROSECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_ts_us>>();
      break;
    case type_id::TIMESTAMP_NANOSECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_ts_ns>>();
      break;
    case type_id::DURATION_DAYS:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_du_day>>();
      break;
    case type_id::DURATION_SECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_du_s>>();
      break;
    case type_id::DURATION_MILLISECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_du_ms>>();
      break;
    case type_id::DURATION_MICROSECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_du_us>>();
      break;
    case type_id::DURATION_NANOSECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_du_ns>>();
      break;
    case type_id::DICTIONARY32:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_dict>>();
      break;
    case type_id::STRING: planner.add_static_fragment<fragment_tag_murmur_hasher<tag_str>>(); break;
    case type_id::LIST: planner.add_static_fragment<fragment_tag_murmur_hasher<tag_list>>(); break;
    case type_id::DECIMAL32:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_dec32>>();
      break;
    case type_id::DECIMAL64:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_dec64>>();
      break;
    case type_id::DECIMAL128:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_dec128>>();
      break;
    case type_id::STRUCT:
      planner.add_static_fragment<fragment_tag_murmur_hasher<tag_struct>>();
      break;
    default: break;
  }
}

inline void add_noop_hasher_fragment(cudf::detail::jit_lto::AlgorithmPlanner& planner, type_id id)
{
  using namespace cudf::hashing::detail::jit_lto;
  switch (id) {
    case type_id::INT8:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_i8>>();
      break;
    case type_id::INT16:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_i16>>();
      break;
    case type_id::INT32:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_i32>>();
      break;
    case type_id::INT64:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_i64>>();
      break;
    case type_id::UINT8:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_u8>>();
      break;
    case type_id::UINT16:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_u16>>();
      break;
    case type_id::UINT32:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_u32>>();
      break;
    case type_id::UINT64:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_u64>>();
      break;
    case type_id::FLOAT32:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_f32>>();
      break;
    case type_id::FLOAT64:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_f64>>();
      break;
    case type_id::BOOL8:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_b8>>();
      break;
    case type_id::TIMESTAMP_DAYS:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_ts_day>>();
      break;
    case type_id::TIMESTAMP_SECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_ts_s>>();
      break;
    case type_id::TIMESTAMP_MILLISECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_ts_ms>>();
      break;
    case type_id::TIMESTAMP_MICROSECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_ts_us>>();
      break;
    case type_id::TIMESTAMP_NANOSECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_ts_ns>>();
      break;
    case type_id::DURATION_DAYS:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_du_day>>();
      break;
    case type_id::DURATION_SECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_du_s>>();
      break;
    case type_id::DURATION_MILLISECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_du_ms>>();
      break;
    case type_id::DURATION_MICROSECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_du_us>>();
      break;
    case type_id::DURATION_NANOSECONDS:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_du_ns>>();
      break;
    case type_id::DICTIONARY32:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_dict>>();
      break;
    case type_id::STRING:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_str>>();
      break;
    case type_id::LIST:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_list>>();
      break;
    case type_id::DECIMAL32:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_dec32>>();
      break;
    case type_id::DECIMAL64:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_dec64>>();
      break;
    case type_id::DECIMAL128:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_dec128>>();
      break;
    case type_id::STRUCT:
      planner.add_static_fragment<fragment_tag_murmur_hasher_noop<tag_struct>>();
      break;
    default: break;
  }
}

}  // namespace

inline std::unique_ptr<column> murmurhash3_x86_32_jit(table_view const& input,
                                                      uint32_t seed,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto output = make_numeric_column(data_type(type_to_id<hash_value_type>()),
                                    input.num_rows(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);

  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  bool const nullable     = has_nulls(input);
  auto const preprocessed = cudf::detail::row::hash::preprocessed_table::create(input, stream);
  table_device_view const input_dv{*preprocessed};

  auto output_view = output->mutable_view();
  auto d_output    = mutable_column_device_view::create(output_view, stream);

  cudf::detail::jit_lto::AlgorithmPlanner planner{"cudf_murmurhash3_x86_32_jit_link_kernel",
                                                  murmur_jit_launcher_cache()};

  using namespace cudf::hashing::detail::jit_lto;
  planner.add_static_fragment<fragment_tag_murmur_entry>();

  std::unordered_set<type_id> logical_types;
  collect_table_logical_types(logical_types, input);
  for (type_id const id : murmur_jit_hasher_type_ids) {
    if (logical_types.count(id) != 0u) {
      add_strong_hasher_fragment(planner, id);
    } else {
      add_noop_hasher_fragment(planner, id);
    }
  }

  auto launcher = planner.get_launcher();

  cudf::detail::grid_1d const grid{input.num_rows(), 256};
  launcher
    ->dispatch<void(cudf::mutable_column_device_view, uint32_t, cudf::table_device_view, bool)>(
      stream.value(),
      dim3(grid.num_blocks),
      dim3(grid.num_threads_per_block),
      0,
      *d_output,
      seed,
      input_dv,
      nullable);

  return output;
}

}  // namespace cudf::hashing::detail
