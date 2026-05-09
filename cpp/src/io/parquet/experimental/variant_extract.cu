/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "variant_path.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/experimental/variant.hpp>
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

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace cudf {
namespace io::parquet::experimental {
namespace {

constexpr int variant_version_v1 = 1;

// Path-step kinds used on-device.
constexpr uint8_t step_kind_name  = 0;
constexpr uint8_t step_kind_index = 1;

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

// Parse an array value header, return the sub-span of the element at `index` (0-based) relative to
// the start of this value. Returns {0, 0} if `val` is not an array, if `index` is out of bounds,
// or if the data is truncated.
//
// Array layout per the Variant spec:
//   byte 0: header (basic_type=3 in low 2 bits; header6 in high 6 bits)
//     header6 bits: (offset_size - 1) in bits 0-1, is_large in bit 2, bits 3-5 unused
//   num_elements: 1 byte if !is_large else 4 bytes (little-endian)
//   offsets: (num_elements + 1) entries, each `offset_size` bytes, relative to the end of offsets
//   values:  concatenated element blobs
__device__ inline field_span device_locate_array_element(uint8_t const* val,
                                                         size_type val_len,
                                                         size_type index)
{
  if (val_len < 1) { return {0, 0}; }
  uint8_t const vm     = val[0];
  int const basic_type = vm & 0x03;
  if (basic_type != 3) { return {0, 0}; }  // not array

  int const header6   = (vm >> 2) & 0x3F;
  int const off_size  = (header6 & 0x03) + 1;
  bool const is_large = ((header6 >> 2) & 0x01) != 0;

  size_type pos         = 1;
  auto const num_elts_r = device_read_uint_le(val, val_len, pos, is_large ? 4 : 1);
  if (!num_elts_r.ok) { return {0, 0}; }
  auto const n = static_cast<size_type>(num_elts_r.value);
  pos += is_large ? 4 : 1;

  if (index < 0 || index >= n) { return {0, 0}; }

  size_type const offs_start = pos;
  size_type const offs_bytes = (n + 1) * off_size;
  if (offs_start + offs_bytes > val_len) { return {0, 0}; }
  size_type const values_base = offs_start + offs_bytes;

  auto const o0 = device_read_uint_le(val, val_len, offs_start + index * off_size, off_size);
  auto const o1 = device_read_uint_le(val, val_len, offs_start + (index + 1) * off_size, off_size);
  if (!o0.ok || !o1.ok) { return {0, 0}; }
  auto const start  = static_cast<size_type>(o0.value);
  auto const endoff = static_cast<size_type>(o1.value);
  if (endoff < start || values_base + endoff > val_len) { return {0, 0}; }
  return {values_base + start, endoff - start};
}

template <typename T>
struct int_decode_result {
  T value;
  bool ok;
};

// Variant primitive ints: basic_type=0, header6 maps INT{8,16,32,64} -> {3,4,5,6}.
template <typename T>
__device__ inline int_decode_result<T> device_decode_int(uint8_t const* enc, size_type len)
{
  static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                  std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>,
                "device_decode_int: unsupported width");
  constexpr int header6_for = std::is_same_v<T, int8_t>    ? 3
                              : std::is_same_v<T, int16_t> ? 4
                              : std::is_same_v<T, int32_t> ? 5
                                                           : 6;
  constexpr int width       = sizeof(T);
  if (len < 1 + width) { return {0, false}; }
  uint8_t const vm = enc[0];
  if ((vm & 0x03) != 0 || ((vm >> 2) & 0x3F) != header6_for) { return {0, false}; }
  using U = std::make_unsigned_t<T>;
  U u     = 0;
  for (int i = 0; i < width; ++i) {
    u |= static_cast<U>(enc[1 + i]) << (8 * i);
  }
  return {static_cast<T>(u), true};
}

// Walk a path (name or index steps) level by level starting at (val, val_len) and return the span
// of the final value relative to the root val pointer. Returns {0, 0} on failure (empty path, key
// absent, index out of bounds, kind mismatch, or truncated data).
__device__ inline field_span device_locate_path(uint8_t const* meta,
                                                size_type meta_len,
                                                uint8_t const* val,
                                                size_type val_len,
                                                uint8_t const* d_step_kinds,
                                                size_type const* d_step_indices,
                                                char const* d_name_bytes,
                                                size_type const* d_name_offsets,
                                                size_type depth)
{
  if (depth <= 0) { return {0, 0}; }

  uint8_t const* sub_val = val;
  size_type sub_len      = val_len;
  size_type abs_offset   = 0;

  for (size_type i = 0; i < depth; ++i) {
    field_span fs{0, 0};
    if (d_step_kinds[i] == step_kind_name) {
      auto const name_begin = d_name_offsets[i];
      auto const name_end   = d_name_offsets[i + 1];
      auto const name_len   = name_end - name_begin;
      auto const* name_ptr  = d_name_bytes + name_begin;

      int const dict_idx = device_find_key_in_metadata(meta, meta_len, name_ptr, name_len);
      if (dict_idx < 0) { return {0, 0}; }

      fs = device_locate_object_field(sub_val, sub_len, dict_idx);
    } else {
      fs = device_locate_array_element(sub_val, sub_len, d_step_indices[i]);
    }
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
                                                uint8_t const* d_step_kinds,
                                                size_type const* d_step_indices,
                                                char const* d_name_bytes,
                                                size_type const* d_name_offsets,
                                                size_type depth,
                                                size_type* d_sizes,
                                                size_type* d_src_offsets,
                                                bitmask_type* d_null_mask,
                                                size_type num_rows)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }
  auto const row = static_cast<size_type>(tid);

  if (d_struct.nullable() && !d_struct.is_valid_nocheck(row)) {
    d_sizes[row]       = 0;
    d_src_offsets[row] = 0;
    cudf::clear_bit(d_null_mask, row);
    return;
  }

  auto const [meta_ptr, meta_len, val_ptr, val_len] = get_row_lists(d_meta, d_val, row);

  auto const fs = device_locate_path(meta_ptr,
                                     meta_len,
                                     val_ptr,
                                     val_len,
                                     d_step_kinds,
                                     d_step_indices,
                                     d_name_bytes,
                                     d_name_offsets,
                                     depth);
  if (fs.length == 0) {
    d_sizes[row]       = 0;
    d_src_offsets[row] = 0;
    cudf::clear_bit(d_null_mask, row);
    return;
  }

  d_sizes[row]       = fs.length;
  d_src_offsets[row] = fs.offset;
}

// Pass 2: pure gather/copy. Source location was memoized by the sizes kernel,
// so this kernel does no Variant parsing — it just copies `d_sizes[row]` bytes
// from the row's value blob (offset by the cached intra-blob offset) to the
// output buffer. The per-row size is recovered from the offsets column as
// `d_offsets[row + 1] - d_offsets[row]`.
CUDF_KERNEL void get_variant_field_copy_kernel(cudf::detail::lists_column_device_view d_val,
                                               size_type const* d_src_offsets,
                                               size_type const* d_offsets,
                                               uint8_t* d_out_bytes,
                                               bitmask_type const* d_null_mask,
                                               size_type num_rows)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }
  auto const row = static_cast<size_type>(tid);

  if (!cudf::bit_is_set(d_null_mask, row)) { return; }

  auto const dst_begin = d_offsets[row];
  auto const len       = d_offsets[row + 1] - dst_begin;
  if (len == 0) { return; }

  auto const val_begin = d_val.offset_at(row);
  auto const* val_ptr  = d_val.child().data<uint8_t>() + val_begin;
  auto const* src      = val_ptr + d_src_offsets[row];
  auto* dst            = d_out_bytes + dst_begin;
  for (size_type i = 0; i < len; ++i) {
    dst[i] = src[i];
  }
}

template <typename T>
CUDF_KERNEL void cast_variant_int_kernel(column_device_view d_struct,
                                         cudf::detail::lists_column_device_view d_val,
                                         T* d_output,
                                         bitmask_type* d_null_mask,
                                         size_type num_rows)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }
  auto const row = static_cast<size_type>(tid);

  if (d_struct.nullable() && !d_struct.is_valid_nocheck(row)) {
    d_output[row] = 0;
    cudf::clear_bit(d_null_mask, row);
    return;
  }

  auto const val_begin = d_val.offset_at(row);
  auto const val_end   = d_val.offset_at(row + 1);
  auto const val_child = d_val.child();
  auto const* val_ptr  = val_child.data<uint8_t>() + val_begin;
  auto const val_len   = val_end - val_begin;

  auto const result = device_decode_int<T>(val_ptr, val_len);
  if (result.ok) {
    d_output[row] = result.value;
  } else {
    d_output[row] = 0;
    cudf::clear_bit(d_null_mask, row);
  }
}

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
      cudf::clear_bit(d_null_mask, row);
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
      cudf::clear_bit(d_null_mask, row);
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

  auto const num_rows = variant_column.size();
  CUDF_EXPECTS(meta_lists.size() == num_rows && val_lists.size() == num_rows,
               "VARIANT metadata and value lists must have the same row count as the parent struct",
               std::invalid_argument);
}

// Marshal a parsed path into the three parallel host-side arrays the kernels expect.
struct marshalled_path {
  std::vector<uint8_t> kinds;
  std::vector<size_type> indices;
  std::string name_bytes;
  std::vector<size_type> name_offsets;
};

marshalled_path marshal_path(std::vector<detail::variant_path_step> const& steps)
{
  auto const depth = static_cast<size_type>(steps.size());
  marshalled_path out;
  out.kinds.resize(depth);
  out.indices.assign(depth, 0);
  out.name_offsets.resize(depth + 1);

  size_type running = 0;
  for (size_type i = 0; i < depth; ++i) {
    out.name_offsets[i] = running;
    if (std::holds_alternative<std::string>(steps[i])) {
      out.kinds[i]     = step_kind_name;
      auto const& name = std::get<std::string>(steps[i]);
      out.name_bytes.append(name);
      running += static_cast<size_type>(name.size());
    } else {
      out.kinds[i] = step_kind_index;
      auto const v = std::get<cudf::size_type>(steps[i]);
      CUDF_EXPECTS(v >= 0, "array index must be non-negative", std::invalid_argument);
      out.indices[i] = v;
    }
  }
  out.name_offsets[depth] = running;
  return out;
}

constexpr size_type block_size = 256;

}  // namespace

namespace detail {

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const steps = parse_variant_path(path);
  CUDF_EXPECTS(
    !steps.empty(), "variant path must contain at least one step", std::invalid_argument);

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

  // Marshal the parsed path into host-side arrays, then copy each to device.
  auto const host_path = marshal_path(steps);
  auto const depth     = static_cast<size_type>(steps.size());

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto d_step_kinds  = cudf::detail::make_device_uvector_async(
    host_span<uint8_t const>{host_path.kinds.data(), host_path.kinds.size()}, stream, temp_mr);
  auto d_step_indices = cudf::detail::make_device_uvector_async(
    host_span<size_type const>{host_path.indices.data(), host_path.indices.size()},
    stream,
    temp_mr);
  auto d_name_bytes = cudf::detail::make_device_uvector_async(
    host_span<char const>{host_path.name_bytes.data(), host_path.name_bytes.size()},
    stream,
    temp_mr);
  auto d_name_offsets = cudf::detail::make_device_uvector_async(
    host_span<size_type const>{host_path.name_offsets.data(), host_path.name_offsets.size()},
    stream,
    temp_mr);

  // Create device views for the struct and list children
  auto d_struct_ptr = column_device_view::create(variant_column, stream);

  lists_column_view const meta_lcv{variant_column.child(0)};
  lists_column_view const val_lcv{variant_column.child(1)};
  auto d_meta_col_ptr = column_device_view::create(meta_lcv.parent(), stream);
  auto d_val_col_ptr  = column_device_view::create(val_lcv.parent(), stream);
  cudf::detail::lists_column_device_view d_meta(*d_meta_col_ptr);
  cudf::detail::lists_column_device_view d_val(*d_val_col_ptr);

  // Allocate sizes array, source-offset memo, and null mask (all-valid initially).
  // d_src_offsets caches the per-row intra-value byte offset returned by
  // device_locate_path so the copy kernel can skip re-parsing.
  rmm::device_uvector<size_type> d_sizes(num_rows, stream);
  rmm::device_uvector<size_type> d_src_offsets(num_rows, stream);
  auto null_mask    = cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  // Pass 1: parse the path per row, compute output sizes, and memoize the
  // intra-blob source offset for the copy kernel.
  auto grid = cudf::detail::grid_1d{num_rows, block_size};
  get_variant_field_sizes_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    *d_struct_ptr,
    d_meta,
    d_val,
    d_step_kinds.data(),
    d_step_indices.data(),
    d_name_bytes.data(),
    d_name_offsets.data(),
    depth,
    d_sizes.data(),
    d_src_offsets.data(),
    d_null_mask,
    num_rows);

  // Convert sizes to offsets
  auto [offsets_column, total_bytes] =
    cudf::strings::detail::make_offsets_child_column(d_sizes.begin(), d_sizes.end(), stream, mr);
  auto const* d_offsets = offsets_column->view().data<size_type>();

  // Pass 2: pure gather/copy using the memoized source offsets.
  auto d_out_bytes = rmm::device_uvector<uint8_t>(total_bytes, stream, mr);
  if (total_bytes > 0) {
    get_variant_field_copy_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
      d_val, d_src_offsets.data(), d_offsets, d_out_bytes.data(), d_null_mask, num_rows);
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

std::unique_ptr<column> cast_variant(column_view const& variant_column,
                                     data_type desired_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  auto const tid = desired_type.id();
  CUDF_EXPECTS(tid == type_id::STRING || tid == type_id::INT8 || tid == type_id::INT16 ||
                 tid == type_id::INT32 || tid == type_id::INT64,
               "cast_variant supports STRING and INT8/INT16/INT32/INT64 only",
               std::invalid_argument);

  size_type const num_rows = variant_column.size();
  if (num_rows == 0) { return cudf::make_empty_column(desired_type); }

  auto d_struct_ptr = column_device_view::create(variant_column, stream);

  lists_column_view const val_lcv{variant_column.child(1)};
  auto d_val_col_ptr = column_device_view::create(val_lcv.parent(), stream);
  cudf::detail::lists_column_device_view d_val(*d_val_col_ptr);

  auto launch_int = [&]<typename T>(std::in_place_type_t<T>) {
    rmm::device_buffer data{static_cast<std::size_t>(num_rows) * sizeof(T), stream, mr};
    auto null_mask    = cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
    auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

    auto grid = cudf::detail::grid_1d{num_rows, block_size};
    cast_variant_int_kernel<T><<<grid.num_blocks, block_size, 0, stream.value()>>>(
      *d_struct_ptr, d_val, static_cast<T*>(data.data()), d_null_mask, num_rows);

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);

    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                                    null_count);
  };

  switch (tid) {
    case type_id::INT8: return launch_int(std::in_place_type<int8_t>);
    case type_id::INT16: return launch_int(std::in_place_type<int16_t>);
    case type_id::INT32: return launch_int(std::in_place_type<int32_t>);
    case type_id::INT64: return launch_int(std::in_place_type<int64_t>);
    default: break;
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

}  // namespace detail

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  validate_variant_struct(variant_column);
  return detail::get_variant_field(variant_column, path, stream, mr);
}

std::unique_ptr<column> cast_variant(column_view const& variant_column,
                                     data_type desired_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  validate_variant_struct(variant_column);
  return detail::cast_variant(variant_column, desired_type, stream, mr);
}

std::unique_ptr<column> extract_variant_field(column_view const& variant_column,
                                              std::string_view path,
                                              data_type desired_type,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  validate_variant_struct(variant_column);
  // Validate once; the intermediate from detail::get_variant_field is always a well-formed
  // VARIANT struct, so detail::cast_variant can skip its own validation.
  auto intermediate = detail::get_variant_field(variant_column, path, stream, mr);
  return detail::cast_variant(intermediate->view(), desired_type, stream, mr);
}

}  // namespace io::parquet::experimental
}  // namespace cudf
