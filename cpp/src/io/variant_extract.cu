/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/variant.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

#include <array>

namespace cudf {
namespace io::parquet {
namespace {

constexpr int variant_version_v1 = 1;

// ---------------------------------------------------------------------------
// Device-side parsing helpers
// ---------------------------------------------------------------------------

struct uint_result {
  uint64_t value;
  bool ok;
};

struct field_span {
  size_type offset;
  size_type length;
};

__device__ inline uint_result device_read_uint_le(uint8_t const* data,
                                                  size_type len,
                                                  size_type pos,
                                                  int width)
{
  if (pos + width > len) { return {0, false}; }
  uint64_t v = 0;
  for (int i = 0; i < width; ++i) {
    v |= static_cast<uint64_t>(data[pos + i]) << (8 * i);
  }
  return {v, true};
}

// Parse metadata header, walk dictionary entries, return the index of `key` or -1.
__device__ inline int device_find_key_in_metadata(uint8_t const* meta,
                                                  size_type meta_len,
                                                  char const* key,
                                                  size_type key_len)
{
  if (meta_len < 1) { return -1; }
  uint8_t const header = meta[0];
  int const version    = header & 0x0F;
  if (version != variant_version_v1) { return -1; }
  int const offset_size = ((header >> 6) & 0x03) + 1;

  size_type pos          = 1;
  auto const dict_size_r = device_read_uint_le(meta, meta_len, pos, offset_size);
  if (!dict_size_r.ok) { return -1; }
  auto const dict_size = static_cast<size_type>(dict_size_r.value);
  pos += offset_size;

  // Read dictionary_size + 1 offsets
  size_type const offsets_start = pos;
  size_type const offsets_bytes = (dict_size + 1) * offset_size;
  if (offsets_start + offsets_bytes > meta_len) { return -1; }

  size_type const strings_base = offsets_start + offsets_bytes;

  // Carry forward the previous offset to avoid re-reading it each iteration.
  auto prev_off_r = device_read_uint_le(meta, meta_len, offsets_start, offset_size);
  if (!prev_off_r.ok) { return -1; }

  for (size_type i = 0; i < dict_size; ++i) {
    auto const next_off_r =
      device_read_uint_le(meta, meta_len, offsets_start + (i + 1) * offset_size, offset_size);
    if (!next_off_r.ok) { return -1; }
    auto const soff = static_cast<size_type>(prev_off_r.value);
    auto const slen = static_cast<size_type>(next_off_r.value) - soff;
    prev_off_r      = next_off_r;
    if (slen != key_len) { continue; }
    if (strings_base + soff + slen > meta_len) { return -1; }
    bool match = true;
    for (size_type c = 0; c < slen; ++c) {
      if (meta[strings_base + soff + c] != static_cast<uint8_t>(key[c])) {
        match = false;
        break;
      }
    }
    if (match) { return i; }
  }
  return -1;
}

// Parse an object value header, find the field with the given dictionary index,
// return the sub-span {offset_from_value_start, length} of the field's encoded value.
// Returns {0, 0} on failure.
//
// Per the Variant spec, field IDs are sorted lexicographically by name but field VALUES
// may be stored in any order, so field_offsets are NOT necessarily monotonically increasing.
// To find a field's byte range we locate the smallest offset strictly greater than the
// field's start offset among all entries (including the sentinel), giving the tightest bound.
__device__ inline field_span device_locate_object_field(uint8_t const* val,
                                                        size_type val_len,
                                                        int dict_idx)
{
  if (val_len < 1) { return {0, 0}; }
  uint8_t const vm     = val[0];
  int const basic_type = vm & 0x03;
  if (basic_type != 2) { return {0, 0}; }  // not object

  int const value_header   = (vm >> 2) & 0x3F;
  int const field_off_size = (value_header & 0x03) + 1;
  int const field_id_size  = ((value_header >> 2) & 0x03) + 1;
  bool const is_large      = ((value_header >> 4) & 0x01) != 0;

  size_type pos         = 1;
  auto const num_elts_r = device_read_uint_le(val, val_len, pos, is_large ? 4 : 1);
  if (!num_elts_r.ok) { return {0, 0}; }
  auto const n = static_cast<size_type>(num_elts_r.value);
  pos += is_large ? 4 : 1;

  size_type const field_ids_start = pos;
  size_type const field_ids_bytes = n * field_id_size;
  if (field_ids_start + field_ids_bytes > val_len) { return {0, 0}; }

  size_type const field_offs_start = field_ids_start + field_ids_bytes;
  size_type const field_offs_bytes = (n + 1) * field_off_size;
  if (field_offs_start + field_offs_bytes > val_len) { return {0, 0}; }

  size_type const values_base = field_offs_start + field_offs_bytes;

  // Find the matching field ID and its start offset.
  size_type match_start = -1;
  for (size_type i = 0; i < n; ++i) {
    auto const fid_r =
      device_read_uint_le(val, val_len, field_ids_start + i * field_id_size, field_id_size);
    if (!fid_r.ok) { return {0, 0}; }
    if (static_cast<int>(fid_r.value) != dict_idx) { continue; }

    auto const o_r =
      device_read_uint_le(val, val_len, field_offs_start + i * field_off_size, field_off_size);
    if (!o_r.ok) { return {0, 0}; }
    match_start = static_cast<size_type>(o_r.value);
    break;
  }
  if (match_start < 0) { return {0, 0}; }

  // Find the tightest end: the smallest offset strictly greater than match_start
  // among all n+1 offset entries (the sentinel at index n is the total data size).
  size_type match_end = val_len - values_base;  // upper bound
  for (size_type j = 0; j <= n; ++j) {
    auto const oj_r =
      device_read_uint_le(val, val_len, field_offs_start + j * field_off_size, field_off_size);
    if (!oj_r.ok) { continue; }
    auto const oj = static_cast<size_type>(oj_r.value);
    if (oj > match_start && oj < match_end) { match_end = oj; }
  }

  if (values_base + match_end > val_len) { return {0, 0}; }
  return {values_base + match_start, match_end - match_start};
}

struct int32_decode_result {
  int32_t value;
  bool ok;
};

__device__ inline int32_decode_result device_decode_int32(uint8_t const* enc, size_type len)
{
  if (len < 5) { return {0, false}; }
  uint8_t const vm     = enc[0];
  int const basic_type = vm & 0x03;
  int const header6    = (vm >> 2) & 0x3F;
  if (basic_type != 0 || header6 != 5) { return {0, false}; }
  uint32_t u = 0;
  for (int i = 0; i < 4; ++i) {
    u |= static_cast<uint32_t>(enc[1 + i]) << (8 * i);
  }
  return {static_cast<int32_t>(u), true};
}

// Walk an object key path level by level starting at (val, val_len) and return the span of the
// final value relative to the root val pointer. Returns {0, 0} on failure (empty path, key absent,
// non-object intermediate, or truncated data).
__device__ inline field_span device_locate_object_path(uint8_t const* meta,
                                                       size_type meta_len,
                                                       uint8_t const* val,
                                                       size_type val_len,
                                                       char const* d_key_bytes,
                                                       size_type const* d_key_offsets,
                                                       size_type depth)
{
  if (depth <= 0) { return {0, 0}; }

  uint8_t const* sub_val = val;
  size_type sub_len      = val_len;
  size_type abs_offset   = 0;

  for (size_type i = 0; i < depth; ++i) {
    auto const key_begin = d_key_offsets[i];
    auto const key_end   = d_key_offsets[i + 1];
    auto const key_len   = key_end - key_begin;
    auto const* key_ptr  = d_key_bytes + key_begin;

    int const dict_idx = device_find_key_in_metadata(meta, meta_len, key_ptr, key_len);
    if (dict_idx < 0) { return {0, 0}; }

    auto const fs = device_locate_object_field(sub_val, sub_len, dict_idx);
    if (fs.length == 0) { return {0, 0}; }

    sub_val += fs.offset;
    abs_offset += fs.offset;
    sub_len = fs.length;
  }

  return {abs_offset, sub_len};
}

struct string_decode_result {
  size_type length;
  size_type data_offset;  // offset from enc[0] to first char byte
  bool ok;
};

__device__ inline string_decode_result device_decode_string_info(uint8_t const* enc, size_type len)
{
  if (len < 1) { return {0, 0, false}; }
  uint8_t const vm     = enc[0];
  int const basic_type = vm & 0x03;
  int const header6    = (vm >> 2) & 0x3F;

  if (basic_type == 1) {
    // Short string: header6 = length
    size_type const slen = header6;
    if (1 + slen > len) { return {0, 0, false}; }
    return {slen, 1, true};
  }
  if (basic_type == 0 && header6 == 16) {
    // Long string: 4-byte LE length follows header
    if (len < 5) { return {0, 0, false}; }
    uint32_t slen = 0;
    for (int i = 0; i < 4; ++i) {
      slen |= static_cast<uint32_t>(enc[1 + i]) << (8 * i);
    }
    if (5 + static_cast<size_type>(slen) > len) { return {0, 0, false}; }
    return {static_cast<size_type>(slen), 5, true};
  }
  return {0, 0, false};
}

// ---------------------------------------------------------------------------
// get_variant_field: two-pass kernel
// ---------------------------------------------------------------------------

// Reads the metadata and value list bytes for a given row from device views.
// Returns pointers into device memory and the lengths.
struct row_list_ptrs {
  uint8_t const* meta_ptr;
  size_type meta_len;
  uint8_t const* val_ptr;
  size_type val_len;
};

__device__ inline row_list_ptrs get_row_lists(cudf::detail::lists_column_device_view const& d_meta,
                                              cudf::detail::lists_column_device_view const& d_val,
                                              size_type row)
{
  auto const meta_begin = d_meta.offset_at(row);
  auto const meta_end   = d_meta.offset_at(row + 1);
  auto const val_begin  = d_val.offset_at(row);
  auto const val_end    = d_val.offset_at(row + 1);
  auto const meta_child = d_meta.child();
  auto const val_child  = d_val.child();
  return {meta_child.data<uint8_t>() + meta_begin,
          meta_end - meta_begin,
          val_child.data<uint8_t>() + val_begin,
          val_end - val_begin};
}

CUDF_KERNEL void get_variant_field_sizes_kernel(column_device_view d_struct,
                                                cudf::detail::lists_column_device_view d_meta,
                                                cudf::detail::lists_column_device_view d_val,
                                                char const* d_key_bytes,
                                                size_type const* d_key_offsets,
                                                size_type depth,
                                                size_type* d_sizes,
                                                bitmask_type* d_null_mask,
                                                size_type num_rows)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }
  auto const row = static_cast<size_type>(tid);

  if (d_struct.nullable() && !d_struct.is_valid_nocheck(row)) {
    d_sizes[row] = 0;
    cudf::clear_bit_unsafe(d_null_mask, row);
    return;
  }

  auto const [meta_ptr, meta_len, val_ptr, val_len] = get_row_lists(d_meta, d_val, row);

  auto const fs = device_locate_object_path(
    meta_ptr, meta_len, val_ptr, val_len, d_key_bytes, d_key_offsets, depth);
  if (fs.length == 0) {
    d_sizes[row] = 0;
    cudf::clear_bit_unsafe(d_null_mask, row);
    return;
  }

  d_sizes[row] = fs.length;
}

CUDF_KERNEL void get_variant_field_copy_kernel(cudf::detail::lists_column_device_view d_meta,
                                               cudf::detail::lists_column_device_view d_val,
                                               char const* d_key_bytes,
                                               size_type const* d_key_offsets,
                                               size_type depth,
                                               size_type const* d_offsets,
                                               uint8_t* d_out_bytes,
                                               bitmask_type const* d_null_mask,
                                               size_type num_rows)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }
  auto const row = static_cast<size_type>(tid);

  if (!cudf::bit_is_set(d_null_mask, row)) { return; }

  auto const [meta_ptr, meta_len, val_ptr, val_len] = get_row_lists(d_meta, d_val, row);

  auto const fs = device_locate_object_path(
    meta_ptr, meta_len, val_ptr, val_len, d_key_bytes, d_key_offsets, depth);
  if (fs.length == 0) { return; }

  auto* dst = d_out_bytes + d_offsets[row];
  for (size_type i = 0; i < fs.length; ++i) {
    dst[i] = val_ptr[fs.offset + i];
  }
}

// ---------------------------------------------------------------------------
// cast_variant: INT32 kernel
// ---------------------------------------------------------------------------

CUDF_KERNEL void cast_variant_int32_kernel(column_device_view d_struct,
                                           cudf::detail::lists_column_device_view d_val,
                                           int32_t* d_output,
                                           bitmask_type* d_null_mask,
                                           size_type num_rows)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }
  auto const row = static_cast<size_type>(tid);

  if (d_struct.nullable() && !d_struct.is_valid_nocheck(row)) {
    d_output[row] = 0;
    cudf::clear_bit_unsafe(d_null_mask, row);
    return;
  }

  auto const val_begin = d_val.offset_at(row);
  auto const val_end   = d_val.offset_at(row + 1);
  auto const val_child = d_val.child();
  auto const* val_ptr  = val_child.data<uint8_t>() + val_begin;
  auto const val_len   = val_end - val_begin;

  auto const result = device_decode_int32(val_ptr, val_len);
  if (result.ok) {
    d_output[row] = result.value;
  } else {
    d_output[row] = 0;
    cudf::clear_bit_unsafe(d_null_mask, row);
  }
}

// ---------------------------------------------------------------------------
// cast_variant: STRING functor for make_strings_children
// ---------------------------------------------------------------------------

struct cast_variant_string_fn {
  column_device_view d_struct;
  cudf::detail::lists_column_device_view d_val;
  bitmask_type* d_null_mask;
  size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type row)
  {
    if (d_struct.nullable() && !d_struct.is_valid_nocheck(row)) {
      if (!d_chars) { d_sizes[row] = 0; }
      cudf::clear_bit_unsafe(d_null_mask, row);
      return;
    }

    auto const val_begin = d_val.offset_at(row);
    auto const val_end   = d_val.offset_at(row + 1);
    auto const val_child = d_val.child();
    auto const* val_ptr  = val_child.data<uint8_t>() + val_begin;
    auto const val_len   = val_end - val_begin;

    auto const info = device_decode_string_info(val_ptr, val_len);
    if (!info.ok) {
      if (!d_chars) { d_sizes[row] = 0; }
      cudf::clear_bit_unsafe(d_null_mask, row);
      return;
    }

    if (!d_chars) {
      d_sizes[row] = info.length;
    } else {
      auto* out = d_chars + d_offsets[row];
      for (size_type i = 0; i < info.length; ++i) {
        out[i] = static_cast<char>(val_ptr[info.data_offset + i]);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Validation helpers shared between public APIs
// ---------------------------------------------------------------------------

void validate_variant_struct(column_view const& variant_column)
{
  CUDF_EXPECTS(variant_column.type().id() == type_id::STRUCT,
               "VARIANT column must be struct type",
               std::invalid_argument);
  CUDF_EXPECTS(
    variant_column.num_children() >= 2,
    "VARIANT struct must start with metadata and value children (list<uint8>) before any optional "
    "shredded columns",
    std::invalid_argument);

  lists_column_view const meta_lists{variant_column.child(0)};
  lists_column_view const val_lists{variant_column.child(1)};
  CUDF_EXPECTS(meta_lists.child().type().id() == type_id::UINT8 &&
                 val_lists.child().type().id() == type_id::UINT8,
               "VARIANT metadata and value children must be list<uint8>",
               std::invalid_argument);
}

constexpr size_type block_size = 256;

}  // namespace

// ---------------------------------------------------------------------------
// get_variant_field
// ---------------------------------------------------------------------------

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          host_span<std::string const> field_path,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  validate_variant_struct(variant_column);
  CUDF_EXPECTS(
    !field_path.empty(), "field_path must contain at least one key", std::invalid_argument);

  size_type const num_rows = variant_column.size();
  if (num_rows == 0) {
    auto meta_child = cudf::make_lists_column(
      0, make_empty_column(type_id::INT32), make_empty_column(type_id::UINT8), 0, {});
    auto val_child = cudf::make_lists_column(
      0, make_empty_column(type_id::INT32), make_empty_column(type_id::UINT8), 0, {});
    std::vector<std::unique_ptr<column>> children;
    children.push_back(std::move(meta_child));
    children.push_back(std::move(val_child));
    return make_structs_column(0, std::move(children), 0, {}, stream, mr);
  }

  // Pack the path into a single concatenated char buffer + offsets array.
  auto const depth = static_cast<size_type>(field_path.size());
  std::vector<size_type> h_key_offsets(depth + 1);
  size_type running = 0;
  for (size_type i = 0; i < depth; ++i) {
    h_key_offsets[i] = running;
    running += static_cast<size_type>(field_path[i].size());
  }
  h_key_offsets[depth] = running;
  std::string h_key_bytes;
  h_key_bytes.reserve(running);
  for (auto const& k : field_path) {
    h_key_bytes.append(k);
  }

  auto d_key_bytes = cudf::detail::make_device_uvector_async(
    host_span<char const>{h_key_bytes.data(), h_key_bytes.size()}, stream, mr);
  auto d_key_offsets = cudf::detail::make_device_uvector_async(
    host_span<size_type const>{h_key_offsets.data(), h_key_offsets.size()}, stream, mr);

  // Create device views for the struct and list children
  auto d_struct_ptr = column_device_view::create(variant_column, stream);

  lists_column_view const meta_lcv{variant_column.child(0)};
  lists_column_view const val_lcv{variant_column.child(1)};
  auto d_meta_col_ptr = column_device_view::create(meta_lcv.parent(), stream);
  auto d_val_col_ptr  = column_device_view::create(val_lcv.parent(), stream);
  cudf::detail::lists_column_device_view d_meta(*d_meta_col_ptr);
  cudf::detail::lists_column_device_view d_val(*d_val_col_ptr);

  // Allocate sizes array and null mask (all-valid initially)
  rmm::device_uvector<size_type> d_sizes(num_rows, stream);
  auto null_mask    = cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  // Pass 1: compute sizes
  auto grid = cudf::detail::grid_1d{num_rows, block_size};
  get_variant_field_sizes_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    *d_struct_ptr,
    d_meta,
    d_val,
    d_key_bytes.data(),
    d_key_offsets.data(),
    depth,
    d_sizes.data(),
    d_null_mask,
    num_rows);

  // Convert sizes to offsets
  auto [offsets_column, total_bytes] =
    cudf::strings::detail::make_offsets_child_column(d_sizes.begin(), d_sizes.end(), stream, mr);
  auto const* d_offsets = offsets_column->view().data<size_type>();

  // Allocate output byte buffer and run pass 2
  auto d_out_bytes = rmm::device_uvector<uint8_t>(total_bytes, stream, mr);
  if (total_bytes > 0) {
    get_variant_field_copy_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
      d_meta,
      d_val,
      d_key_bytes.data(),
      d_key_offsets.data(),
      depth,
      d_offsets,
      d_out_bytes.data(),
      d_null_mask,
      num_rows);
  }

  // Build the output value list<uint8> column
  auto uint8_child = std::make_unique<column>(data_type{type_id::UINT8},
                                              static_cast<size_type>(total_bytes),
                                              d_out_bytes.release(),
                                              rmm::device_buffer{},
                                              0);
  auto val_out =
    make_lists_column(num_rows, std::move(offsets_column), std::move(uint8_child), 0, {});

  // Copy input metadata column for the output
  auto meta_out = std::make_unique<column>(variant_column.child(0), stream, mr);

  // Compute null count
  auto const null_count = num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);

  // Assemble struct without propagating nulls into children — the struct-level
  // null mask is authoritative.  Skipping superimpose_and_sanitize_nulls avoids
  // an expensive purge_nonempty_nulls deep-copy of the metadata list column.
  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(meta_out));
  children.push_back(std::move(val_out));
  return create_structs_hierarchy(num_rows,
                                  std::move(children),
                                  null_count,
                                  null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                                  stream,
                                  mr);
}

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string const& field_name,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  std::array<std::string, 1> path{field_name};
  return get_variant_field(
    variant_column, host_span<std::string const>{path.data(), path.size()}, stream, mr);
}

// ---------------------------------------------------------------------------
// cast_variant
// ---------------------------------------------------------------------------

std::unique_ptr<column> cast_variant(column_view const& variant_column,
                                     data_type desired_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  validate_variant_struct(variant_column);

  CUDF_EXPECTS(desired_type.id() == type_id::STRING || desired_type.id() == type_id::INT32,
               "cast_variant supports STRING and INT32 only",
               std::invalid_argument);

  size_type const num_rows = variant_column.size();
  if (num_rows == 0) { return cudf::make_empty_column(desired_type); }

  auto d_struct_ptr = column_device_view::create(variant_column, stream);

  lists_column_view const val_lcv{variant_column.child(1)};
  auto d_val_col_ptr = column_device_view::create(val_lcv.parent(), stream);
  cudf::detail::lists_column_device_view d_val(*d_val_col_ptr);

  if (desired_type.id() == type_id::INT32) {
    rmm::device_buffer data{static_cast<std::size_t>(num_rows) * sizeof(int32_t), stream, mr};
    auto null_mask    = cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
    auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

    auto grid = cudf::detail::grid_1d{num_rows, block_size};
    cast_variant_int32_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
      *d_struct_ptr, d_val, static_cast<int32_t*>(data.data()), d_null_mask, num_rows);

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);

    return std::make_unique<column>(data_type{type_id::INT32},
                                    num_rows,
                                    std::move(data),
                                    null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                                    null_count);
  }

  // STRING path
  auto null_mask    = cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  cast_variant_string_fn fn{*d_struct_ptr, d_val, d_null_mask, nullptr, nullptr, {}};
  auto [offsets_column, chars] =
    cudf::strings::detail::make_strings_children(fn, num_rows, stream, mr);

  auto const null_count = num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);

  return make_strings_column(num_rows,
                             std::move(offsets_column),
                             chars.release(),
                             null_count,
                             null_count > 0 ? std::move(null_mask) : rmm::device_buffer{});
}

// ---------------------------------------------------------------------------
// extract_variant_field (convenience wrapper)
// ---------------------------------------------------------------------------

std::unique_ptr<column> extract_variant_field(column_view const& variant_column,
                                              host_span<std::string const> field_path,
                                              data_type desired_type,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto intermediate = get_variant_field(variant_column, field_path, stream, mr);
  return cast_variant(intermediate->view(), desired_type, stream, mr);
}

std::unique_ptr<column> extract_variant_field(column_view const& variant_column,
                                              std::string const& field_name,
                                              data_type desired_type,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  std::array<std::string, 1> path{field_name};
  return extract_variant_field(variant_column,
                               host_span<std::string const>{path.data(), path.size()},
                               desired_type,
                               stream,
                               mr);
}

}  // namespace io::parquet
}  // namespace cudf
