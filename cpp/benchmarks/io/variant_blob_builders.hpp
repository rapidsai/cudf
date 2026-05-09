/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudf::io::parquet::benchmark_util {

// ============================================================================
// Variant binary blob builders (host-side, spec-compliant for the unshredded
// metadata + value layout we use in tests and benchmarks).
//
// All builders use offset_size=1/field_id_size=1/field_off_size=1 when the
// data fits in a byte; the metadata builder widens to 2-byte offsets when the
// concatenated string length exceeds 255.  None of the builders support the
// 4-byte / large variants of every field individually - they are meant for
// benchmark fixtures of moderate size, not as a general-purpose encoder.
// ============================================================================

// ---- Primitives -----------------------------------------------------------

// Short/long-string primitive: short form when `s.size() <= 63`, otherwise the
// 4-byte-length "long string" form (basic_type=0, header6=16).
inline std::vector<uint8_t> build_bare_string_value(std::string_view s)
{
  if (s.size() <= 63) {
    std::vector<uint8_t> out{static_cast<uint8_t>(0x01 | (s.size() << 2))};
    out.insert(out.end(), s.begin(), s.end());
    return out;
  }
  std::vector<uint8_t> out{0x40};  // (16 << 2) | 0 = 0x40
  auto const len = static_cast<uint32_t>(s.size());
  for (int b = 0; b < 4; ++b) {
    out.push_back(static_cast<uint8_t>((len >> (8 * b)) & 0xFF));
  }
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

// Bare INT32 primitive blob (no object wrapping).
inline std::vector<uint8_t> build_bare_int32_value(int32_t v)
{
  auto const u = static_cast<uint32_t>(v);
  return {0x14,
          static_cast<uint8_t>(u & 0xFF),
          static_cast<uint8_t>((u >> 8) & 0xFF),
          static_cast<uint8_t>((u >> 16) & 0xFF),
          static_cast<uint8_t>((u >> 24) & 0xFF)};
}

// ---- Metadata -------------------------------------------------------------

// Build a Variant metadata blob from an explicit list of key-name strings.
// The Variant spec requires the dictionary to be sorted lexicographically;
// callers are responsible for sorting `keys` themselves.
inline std::vector<uint8_t> build_metadata(std::vector<std::string> const& keys)
{
  auto const num_keys = static_cast<int>(keys.size());

  std::size_t total_str_bytes = 0;
  for (auto const& k : keys) {
    total_str_bytes += k.size();
  }
  int const offset_size = (total_str_bytes <= 255) ? 1 : 2;
  uint8_t const header  = 0x01 | (static_cast<uint8_t>(offset_size - 1) << 6);

  std::vector<uint8_t> out{header};

  for (int b = 0; b < offset_size; ++b) {
    out.push_back(static_cast<uint8_t>((num_keys >> (8 * b)) & 0xFF));
  }

  std::size_t running = 0;
  for (int i = 0; i <= num_keys; ++i) {
    for (int b = 0; b < offset_size; ++b) {
      out.push_back(static_cast<uint8_t>((running >> (8 * b)) & 0xFF));
    }
    if (i < num_keys) { running += keys[i].size(); }
  }

  for (auto const& k : keys) {
    out.insert(out.end(), k.begin(), k.end());
  }
  return out;
}

// ---- Composite -----------------------------------------------------------

// Build a Variant object value blob where every field carries a bare INT32
// payload.  `field_ids` must be sorted; if `skip_id >= 0`, that ID is omitted.
inline std::vector<uint8_t> build_object_value(std::vector<int> const& field_ids, int skip_id = -1)
{
  std::vector<int> ids;
  ids.reserve(field_ids.size());
  for (int fid : field_ids) {
    if (fid != skip_id) { ids.push_back(fid); }
  }
  int const n      = static_cast<int>(ids.size());
  int const max_id = ids.empty() ? 0 : *std::max_element(ids.begin(), ids.end());

  int const value_bytes_per_field = 5;
  int const total_value_bytes     = n * value_bytes_per_field;

  int const field_id_size = (max_id <= 255) ? 1 : 2;
  int const field_off_size =
    (total_value_bytes <= 255) ? 1 : ((total_value_bytes <= 65535) ? 2 : 4);
  bool const is_large = (n > 255);

  uint8_t const header6 = static_cast<uint8_t>((field_off_size - 1) | ((field_id_size - 1) << 2) |
                                               (is_large ? (1 << 4) : 0));
  uint8_t const header_byte = static_cast<uint8_t>((header6 << 2) | 2);

  std::vector<uint8_t> out{header_byte};

  if (is_large) {
    for (int b = 0; b < 4; ++b) {
      out.push_back(static_cast<uint8_t>((n >> (8 * b)) & 0xFF));
    }
  } else {
    out.push_back(static_cast<uint8_t>(n));
  }

  for (int fid : ids) {
    for (int b = 0; b < field_id_size; ++b) {
      out.push_back(static_cast<uint8_t>((fid >> (8 * b)) & 0xFF));
    }
  }

  for (int i = 0; i <= n; ++i) {
    int off = i * value_bytes_per_field;
    for (int b = 0; b < field_off_size; ++b) {
      out.push_back(static_cast<uint8_t>((off >> (8 * b)) & 0xFF));
    }
  }

  for (int fid : ids) {
    out.push_back(0x14);
    for (int b = 0; b < 4; ++b) {
      out.push_back(static_cast<uint8_t>((fid >> (8 * b)) & 0xFF));
    }
  }

  return out;
}

// Build a Variant object blob from sorted `field_ids_sorted` where one specific
// field `descent_fid` carries an arbitrary `inner_value` blob, and all other
// fields carry a bare INT32(0).
inline std::vector<uint8_t> build_mixed_object_value(std::vector<int> const& field_ids_sorted,
                                                     int descent_fid,
                                                     std::vector<uint8_t> const& inner_value)
{
  int const n      = static_cast<int>(field_ids_sorted.size());
  int const max_id = n ? field_ids_sorted.back() : 0;

  std::vector<std::vector<uint8_t>> values(n);
  std::size_t total_value_bytes = 0;
  for (int i = 0; i < n; ++i) {
    values[i] = (field_ids_sorted[i] == descent_fid) ? inner_value : build_bare_int32_value(0);
    total_value_bytes += values[i].size();
  }

  int const field_id_size = (max_id <= 255) ? 1 : 2;
  int const field_off_size =
    (total_value_bytes <= 255) ? 1 : ((total_value_bytes <= 65535) ? 2 : 4);
  bool const is_large = (n > 255);

  uint8_t const header6 = static_cast<uint8_t>((field_off_size - 1) | ((field_id_size - 1) << 2) |
                                               (is_large ? (1 << 4) : 0));
  uint8_t const header_byte = static_cast<uint8_t>((header6 << 2) | 2);

  std::vector<uint8_t> out{header_byte};

  if (is_large) {
    for (int b = 0; b < 4; ++b) {
      out.push_back(static_cast<uint8_t>((n >> (8 * b)) & 0xFF));
    }
  } else {
    out.push_back(static_cast<uint8_t>(n));
  }

  for (int fid : field_ids_sorted) {
    for (int b = 0; b < field_id_size; ++b) {
      out.push_back(static_cast<uint8_t>((fid >> (8 * b)) & 0xFF));
    }
  }

  std::size_t running = 0;
  for (int i = 0; i <= n; ++i) {
    for (int b = 0; b < field_off_size; ++b) {
      out.push_back(static_cast<uint8_t>((running >> (8 * b)) & 0xFF));
    }
    if (i < n) { running += values[i].size(); }
  }

  for (auto const& v : values) {
    out.insert(out.end(), v.begin(), v.end());
  }
  return out;
}

// Build `depth` levels of object nesting using a dictionary of n_fields fields
// (0..n_fields-1).  At every level the descent field is `descent_fid` and
// carries the next inner level; the innermost value is a bare INT32(0).
inline std::vector<uint8_t> build_nested_object_value(int n_fields, int depth, int descent_fid)
{
  std::vector<int> fids(n_fields);
  for (int i = 0; i < n_fields; ++i) {
    fids[i] = i;
  }
  auto current_value = build_bare_int32_value(0);
  for (int level = 0; level < depth; ++level) {
    current_value = build_mixed_object_value(fids, descent_fid, current_value);
  }
  return current_value;
}

// Build a Variant array value from concatenated element blobs.  Uses
// off_size=1, is_large=false.  Total element bytes must be < 256; if a larger
// array is needed the caller should fall back to a 2- or 4-byte offset size.
inline std::vector<uint8_t> build_array_value(std::vector<std::vector<uint8_t>> const& elements)
{
  std::vector<uint8_t> out{0x03, static_cast<uint8_t>(elements.size())};

  uint8_t running = 0;
  out.push_back(0x00);
  for (auto const& e : elements) {
    running = static_cast<uint8_t>(running + e.size());
    out.push_back(running);
  }

  for (auto const& e : elements) {
    out.insert(out.end(), e.begin(), e.end());
  }
  return out;
}

// Build a Variant object by looking up field names in `dict`.  `named_fields`
// provides {field_name, value_blob} pairs; the returned object encodes the
// corresponding field IDs in sorted order as required by the spec.  Throws if
// any field name is not present in the dictionary.
inline std::vector<uint8_t> build_named_object_value(
  std::vector<std::string> const& dict,
  std::vector<std::pair<std::string, std::vector<uint8_t>>> const& named_fields)
{
  // Resolve names -> field_ids.
  std::vector<std::pair<int, std::vector<uint8_t>>> by_id;
  by_id.reserve(named_fields.size());
  for (auto const& [name, val] : named_fields) {
    auto it = std::find(dict.begin(), dict.end(), name);
    if (it == dict.end()) {
      throw std::runtime_error("build_named_object_value: missing dictionary key '" + name + "'");
    }
    by_id.emplace_back(static_cast<int>(it - dict.begin()), val);
  }

  std::sort(
    by_id.begin(), by_id.end(), [](auto const& a, auto const& b) { return a.first < b.first; });

  int const n      = static_cast<int>(by_id.size());
  int const max_id = n ? by_id.back().first : 0;

  std::size_t total_value_bytes = 0;
  for (auto const& [fid, val] : by_id) {
    total_value_bytes += val.size();
  }

  int const field_id_size = (max_id <= 255) ? 1 : 2;
  int const field_off_size =
    (total_value_bytes <= 255) ? 1 : ((total_value_bytes <= 65535) ? 2 : 4);
  bool const is_large = (n > 255);

  uint8_t const header6 = static_cast<uint8_t>((field_off_size - 1) | ((field_id_size - 1) << 2) |
                                               (is_large ? (1 << 4) : 0));
  uint8_t const header_byte = static_cast<uint8_t>((header6 << 2) | 2);

  std::vector<uint8_t> out{header_byte};

  if (is_large) {
    for (int b = 0; b < 4; ++b) {
      out.push_back(static_cast<uint8_t>((n >> (8 * b)) & 0xFF));
    }
  } else {
    out.push_back(static_cast<uint8_t>(n));
  }

  for (auto const& [fid, val] : by_id) {
    for (int b = 0; b < field_id_size; ++b) {
      out.push_back(static_cast<uint8_t>((fid >> (8 * b)) & 0xFF));
    }
  }

  std::size_t running = 0;
  for (int i = 0; i <= n; ++i) {
    for (int b = 0; b < field_off_size; ++b) {
      out.push_back(static_cast<uint8_t>((running >> (8 * b)) & 0xFF));
    }
    if (i < n) { running += by_id[i].second.size(); }
  }

  for (auto const& [fid, val] : by_id) {
    out.insert(out.end(), val.begin(), val.end());
  }
  return out;
}

// ---- Column construction -------------------------------------------------

// Wrap `meta_rows` and `val_rows` (one blob per row) into a single VARIANT
// struct column: struct<list<uint8> metadata, list<uint8> value>.
inline std::unique_ptr<cudf::column> build_variant_column(
  std::vector<std::vector<uint8_t>> const& meta_rows,
  std::vector<std::vector<uint8_t>> const& val_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = static_cast<cudf::size_type>(meta_rows.size());

  auto build_list_col = [&](std::vector<std::vector<uint8_t>> const& rows) {
    std::vector<int32_t> h_offsets(num_rows + 1);
    h_offsets[0] = 0;
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      h_offsets[i + 1] = h_offsets[i] + static_cast<int32_t>(rows[i].size());
    }
    int32_t const total = h_offsets[num_rows];

    std::vector<uint8_t> h_data(total);
    for (cudf::size_type i = 0; i < num_rows; ++i) {
      std::copy(rows[i].begin(), rows[i].end(), h_data.begin() + h_offsets[i]);
    }

    rmm::device_buffer d_offsets_buf(h_offsets.size() * sizeof(int32_t), stream, mr);
    cudf::detail::cuda_memcpy(
      cudf::device_span<int32_t>{static_cast<int32_t*>(d_offsets_buf.data()), h_offsets.size()},
      cudf::host_span<int32_t const>{h_offsets.data(), h_offsets.size()},
      stream);

    rmm::device_buffer d_data_buf(h_data.size(), stream, mr);
    if (!h_data.empty()) {
      cudf::detail::cuda_memcpy(
        cudf::device_span<uint8_t>{static_cast<uint8_t*>(d_data_buf.data()), h_data.size()},
        cudf::host_span<uint8_t const>{h_data.data(), h_data.size()},
        stream);
    }

    auto offsets_col =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                     static_cast<cudf::size_type>(h_offsets.size()),
                                     std::move(d_offsets_buf),
                                     rmm::device_buffer{},
                                     0);
    auto data_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT8}, total, std::move(d_data_buf), rmm::device_buffer{}, 0);

    return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(data_col), 0, {});
  };

  auto meta_col = build_list_col(meta_rows);
  auto val_col  = build_list_col(val_rows);

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(std::move(meta_col));
  children.push_back(std::move(val_col));
  return cudf::make_structs_column(num_rows, std::move(children), 0, {}, stream, mr);
}

}  // namespace cudf::io::parquet::benchmark_util
