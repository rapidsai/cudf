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
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace cudf {
namespace io::parquet::experimental {
namespace {

constexpr int variant_version_v1 = 1;

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

// Parse metadata header, walk dictionary entries, return the index of `key` or -1
//
// All offset arithmetic widens to uint64_t before any narrowing cast so a
// malformed metadata blob with offsets >= 2^31 cannot wrap a `size_type`
// past the bounds checks
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
  // Bytes available for dictionary string payloads
  auto const strings_extent = static_cast<uint64_t>(meta_len - strings_base);

  // Carry forward the previous offset to avoid re-reading it each iteration
  auto prev_off_r = device_read_uint_le(meta, meta_len, offsets_start, offset_size);
  if (!prev_off_r.ok) { return -1; }

  for (size_type i = 0; i < dict_size; ++i) {
    auto const next_off_r =
      device_read_uint_le(meta, meta_len, offsets_start + (i + 1) * offset_size, offset_size);
    if (!next_off_r.ok) { return -1; }
    auto const prev_u = prev_off_r.value;
    auto const next_u = next_off_r.value;
    prev_off_r        = next_off_r;
    if (next_u < prev_u || next_u > strings_extent) { return -1; }
    auto const slen_u = next_u - prev_u;
    if (slen_u != static_cast<uint64_t>(key_len)) { continue; }
    auto const soff = static_cast<size_type>(prev_u);
    auto const slen = static_cast<size_type>(slen_u);
    bool match      = true;
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
// return the sub-span {offset_from_value_start, length} of the field's encoded value
// Returns {0, 0} on failure
//
// Per the Variant spec, field IDs are sorted lexicographically by name but field VALUES
// may be stored in any order, so field_offsets are NOT necessarily monotonically increasing
// To find a field's byte range we locate the smallest offset strictly greater than the
// field's start offset among all entries (including the sentinel), giving the tightest bound
//
// All offset values read from `val` are kept in `uint64_t` until we have validated that
// they fit within the value-data extent; this prevents a malformed blob with field
// offsets >= 2^31 from wrapping `size_type` past the bounds checks
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
  // Maximum legitimate field-offset value: bytes available after values_base
  auto const values_extent = static_cast<uint64_t>(val_len - values_base);

  // Find the matching field ID and its start offset
  bool found             = false;
  uint64_t match_start_u = 0;
  for (size_type i = 0; i < n; ++i) {
    auto const fid_r =
      device_read_uint_le(val, val_len, field_ids_start + i * field_id_size, field_id_size);
    if (!fid_r.ok) { return {0, 0}; }
    if (static_cast<int>(fid_r.value) != dict_idx) { continue; }

    auto const o_r =
      device_read_uint_le(val, val_len, field_offs_start + i * field_off_size, field_off_size);
    if (!o_r.ok) { return {0, 0}; }
    if (o_r.value > values_extent) { return {0, 0}; }
    match_start_u = o_r.value;
    found         = true;
    break;
  }
  if (!found) { return {0, 0}; }

  // Find the tightest end: the smallest offset strictly greater than match_start
  // among all n+1 offset entries (the sentinel at index n is the total data size)
  uint64_t match_end_u = values_extent;
  for (size_type j = 0; j <= n; ++j) {
    auto const oj_r =
      device_read_uint_le(val, val_len, field_offs_start + j * field_off_size, field_off_size);
    if (!oj_r.ok) { continue; }
    if (oj_r.value > values_extent) { continue; }
    if (oj_r.value > match_start_u && oj_r.value < match_end_u) { match_end_u = oj_r.value; }
  }

  if (match_end_u < match_start_u) { return {0, 0}; }
  return {values_base + static_cast<size_type>(match_start_u),
          static_cast<size_type>(match_end_u - match_start_u)};
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
  static_assert(cuda::std::is_same_v<T, int8_t> || cuda::std::is_same_v<T, int16_t> ||
                  cuda::std::is_same_v<T, int32_t> || cuda::std::is_same_v<T, int64_t>,
                "device_decode_int: unsupported width");
  constexpr int header6_for = cuda::std::is_same_v<T, int8_t>    ? 3
                              : cuda::std::is_same_v<T, int16_t> ? 4
                              : cuda::std::is_same_v<T, int32_t> ? 5
                                                                 : 6;
  constexpr int width       = sizeof(T);
  if (len < 1 + width) { return {0, false}; }
  uint8_t const vm = enc[0];
  if ((vm & 0x03) != 0 || ((vm >> 2) & 0x3F) != header6_for) { return {0, false}; }
  using U = cuda::std::make_unsigned_t<T>;
  U u     = 0;
  for (int i = 0; i < width; ++i) {
    u |= static_cast<U>(enc[1 + i]) << (8 * i);
  }
  return {static_cast<T>(u), true};
}

// Walk a path of object-key steps level by level starting at (val, val_len) and return the span
// of the final value relative to the root val pointer. Returns {0, 0} on failure (empty path, key
// absent, kind mismatch, or truncated data)
__device__ inline field_span device_locate_path(uint8_t const* meta,
                                                size_type meta_len,
                                                uint8_t const* val,
                                                size_type val_len,
                                                column_device_view path_device_view)
{
  auto const depth = path_device_view.size();
  if (depth <= 0) { return {0, 0}; }

  uint8_t const* sub_val = val;
  size_type sub_len      = val_len;
  size_type abs_offset   = 0;

  for (size_type i = 0; i < depth; ++i) {
    auto const name = path_device_view.element<cudf::string_view>(i);

    int const dict_idx =
      device_find_key_in_metadata(meta, meta_len, name.data(), name.size_bytes());
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

// Reads the metadata and value list bytes for a given row from device views
// Returns pointers into device memory and the lengths
struct row_list_ptrs {
  uint8_t const* meta_ptr;
  size_type meta_len;
  uint8_t const* val_ptr;
  size_type val_len;
};

__device__ inline row_list_ptrs get_row_lists(
  cudf::detail::lists_column_device_view const& metadata,
  cudf::detail::lists_column_device_view const& values,
  size_type row)
{
  auto const meta_begin = metadata.offset_at(row);
  auto const meta_end   = metadata.offset_at(row + 1);
  auto const val_begin  = values.offset_at(row);
  auto const val_end    = values.offset_at(row + 1);
  auto const meta_child = metadata.child();
  auto const val_child  = values.child();
  return {meta_child.data<uint8_t>() + meta_begin,
          meta_end - meta_begin,
          val_child.data<uint8_t>() + val_begin,
          val_end - val_begin};
}

constexpr int block_size = 256;

CUDF_KERNEL __launch_bounds__(block_size) void get_variant_field_sizes_kernel(
  cudf::detail::lists_column_device_view metadata,
  cudf::detail::lists_column_device_view values,
  column_device_view path_device_view,
  device_span<size_type> d_sizes,
  device_span<size_type> d_src_offsets,
  bitmask_type* d_null_mask)
{
  auto const num_rows = static_cast<size_type>(d_sizes.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }
  auto const row = static_cast<size_type>(tid);

  if (!cudf::bit_is_set(d_null_mask, row)) {
    d_sizes[row]       = 0;
    d_src_offsets[row] = 0;
    return;
  }

  auto const [meta_ptr, meta_len, val_ptr, val_len] = get_row_lists(metadata, values, row);

  auto const fs = device_locate_path(meta_ptr, meta_len, val_ptr, val_len, path_device_view);
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
// `d_offsets[row + 1] - d_offsets[row]`
// TODO: pass-2 is a pure gather, so the per-thread byte loop here can be
// replaced with a cooperative copy (warp- or block-per-row, e.g
// `cuda::memcpy_async`) once this path becomes a measured hot spot
CUDF_KERNEL __launch_bounds__(block_size) void get_variant_field_copy_kernel(
  cudf::detail::lists_column_device_view values,
  device_span<size_type const> d_src_offsets,
  device_span<size_type const> d_offsets,
  device_span<uint8_t> d_out_bytes,
  bitmask_type const* d_null_mask)
{
  auto const num_rows = static_cast<size_type>(d_src_offsets.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }
  auto const row = static_cast<size_type>(tid);

  if (!cudf::bit_is_set(d_null_mask, row)) { return; }

  auto const dst_begin = d_offsets[row];
  auto const len       = d_offsets[row + 1] - dst_begin;
  if (len == 0) { return; }

  auto const val_begin = values.offset_at(row);
  auto const* val_ptr  = values.child().data<uint8_t>() + val_begin;
  auto const* src      = val_ptr + d_src_offsets[row];
  auto* dst            = d_out_bytes.data() + dst_begin;
  for (size_type i = 0; i < len; ++i) {
    dst[i] = src[i];
  }
}

template <typename T>
CUDF_KERNEL __launch_bounds__(block_size) void cast_variant_int_kernel(
  cudf::detail::lists_column_device_view values, device_span<T> d_output, bitmask_type* d_null_mask)
{
  auto const num_rows = static_cast<size_type>(d_output.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }
  auto const row = static_cast<size_type>(tid);

  if (!cudf::bit_is_set(d_null_mask, row)) {
    d_output[row] = 0;
    return;
  }

  auto const val_begin = values.offset_at(row);
  auto const val_end   = values.offset_at(row + 1);
  auto const val_child = values.child();
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
  cudf::detail::lists_column_device_view d_values;
  bitmask_type* d_null_mask;
  size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type row)
  {
    if (!cudf::bit_is_set(d_null_mask, row)) {
      if (!d_chars) { d_sizes[row] = 0; }
      return;
    }

    auto const val_begin = d_values.offset_at(row);
    auto const val_end   = d_values.offset_at(row + 1);
    auto const val_child = d_values.child();
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

  // Use raw children for type checks; sliced-row-count alignment is handled implicitly
  // by `structs_column_view::get_sliced_child` everywhere we actually walk the data
  CUDF_EXPECTS(variant_column.child(0).type().id() == type_id::LIST &&
                 variant_column.child(1).type().id() == type_id::LIST,
               "VARIANT metadata and value children must be lists",
               std::invalid_argument);
  lists_column_view const meta_lists{variant_column.child(0)};
  lists_column_view const val_lists{variant_column.child(1)};
  CUDF_EXPECTS(meta_lists.child().type().id() == type_id::UINT8 &&
                 val_lists.child().type().id() == type_id::UINT8,
               "VARIANT metadata and value children must be list<uint8>",
               std::invalid_argument);
}

std::unique_ptr<column> build_path_column(std::vector<std::string> const& steps,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const depth = static_cast<size_type>(steps.size());

  std::string host_chars;
  std::vector<size_type> host_offsets(depth + 1);
  for (size_type i = 0; i < depth; ++i) {
    host_offsets[i] = static_cast<size_type>(host_chars.size());
    host_chars.append(steps[i]);
  }
  host_offsets[depth] = static_cast<size_type>(host_chars.size());

  auto d_offsets   = cudf::detail::make_device_uvector_async(host_offsets, stream, mr);
  auto offsets_col = std::make_unique<column>(data_type{type_id::INT32},
                                              static_cast<size_type>(host_offsets.size()),
                                              d_offsets.release(),
                                              rmm::device_buffer{},
                                              0);

  auto d_chars = cudf::detail::make_device_uvector_async(
    host_span<char const>{host_chars.data(), host_chars.size()}, stream, mr);
  return cudf::make_strings_column(
    depth, std::move(offsets_col), d_chars.release(), 0, rmm::device_buffer{});
}

}  // namespace

namespace detail {

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  // Validate the path even for empty input columns
  auto const steps = parse_variant_path(path);

  auto const num_rows = variant_column.size();
  if (num_rows == 0) {
    // Creating manually because empty_like would include shredded columns
    auto meta_child = cudf::make_lists_column(
      0, make_empty_column(type_id::INT32), make_empty_column(type_id::UINT8), 0, {});
    auto val_child = cudf::make_lists_column(
      0, make_empty_column(type_id::INT32), make_empty_column(type_id::UINT8), 0, {});
    std::vector<std::unique_ptr<column>> children;
    children.push_back(std::move(meta_child));
    children.push_back(std::move(val_child));
    return make_structs_column(0, std::move(children), 0, {}, stream, mr);
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();

  auto path_column      = build_path_column(steps, stream, temp_mr);
  auto path_device_view = column_device_view::create(path_column->view(), stream);

  // Resolve children with respect to any slice/offset on the parent struct
  structs_column_view const variant_struct{variant_column};
  auto const meta_view = variant_struct.get_sliced_child(0, stream);
  auto const val_view  = variant_struct.get_sliced_child(1, stream);

  CUDF_EXPECTS(meta_view.type().id() == type_id::LIST && val_view.type().id() == type_id::LIST,
               "VARIANT metadata and value children must be lists",
               std::invalid_argument);
  CUDF_EXPECTS(lists_column_view{meta_view}.child().type().id() == type_id::UINT8 &&
                 lists_column_view{val_view}.child().type().id() == type_id::UINT8,
               "VARIANT metadata and value children must be list<uint8>",
               std::invalid_argument);

  auto meta_device_view = column_device_view::create(meta_view, stream);
  auto val_device_view  = column_device_view::create(val_view, stream);
  cudf::detail::lists_column_device_view meta_lists_device_view(*meta_device_view);
  cudf::detail::lists_column_device_view val_lists_device_view(*val_device_view);

  rmm::device_uvector<size_type> d_sizes(num_rows, stream, temp_mr);
  // Caches the per-row intra-value byte offset
  rmm::device_uvector<size_type> d_src_offsets(num_rows, stream, temp_mr);
  auto null_mask =
    variant_column.nullable()
      ? cudf::detail::copy_bitmask(variant_column, stream, mr)
      : cudf::create_null_mask(variant_column.size(), mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  // Pass 1: parse the path per row and compute output sizes
  auto grid = cudf::detail::grid_1d{num_rows, block_size};
  get_variant_field_sizes_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    meta_lists_device_view,
    val_lists_device_view,
    *path_device_view,
    d_sizes,
    d_src_offsets,
    d_null_mask);

  // Convert sizes to offsets
  auto [offsets_column, total_bytes] =
    cudf::strings::detail::make_offsets_child_column(d_sizes.begin(), d_sizes.end(), stream, mr);
  CUDF_EXPECTS(total_bytes <= std::numeric_limits<size_type>::max(),
               "VARIANT extracted bytes exceed cudf size_type limit",
               std::overflow_error);
  device_span<size_type const> d_offsets{offsets_column->view().data<size_type>(),
                                         static_cast<std::size_t>(num_rows + 1)};

  // Pass 2: gather/copy directly into the output value child buffer
  auto val_child = make_numeric_column(data_type{type_id::UINT8},
                                       static_cast<size_type>(total_bytes),
                                       mask_state::UNALLOCATED,
                                       stream,
                                       mr);
  if (total_bytes > 0) {
    device_span<uint8_t> d_out_bytes{val_child->mutable_view().data<uint8_t>(),
                                     static_cast<std::size_t>(total_bytes)};
    get_variant_field_copy_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
      val_lists_device_view, d_src_offsets, d_offsets, d_out_bytes, d_null_mask);
  }

  auto val_out =
    make_lists_column(num_rows, std::move(offsets_column), std::move(val_child), 0, {});

  // Copy input metadata column for the output (deep copy materializes any slice)
  auto meta_out = std::make_unique<column>(meta_view, stream, mr);

  auto const null_count = num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);

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

  structs_column_view const variant_struct{variant_column};
  auto const val_view = variant_struct.get_sliced_child(1, stream);
  CUDF_EXPECTS(val_view.type().id() == type_id::LIST,
               "VARIANT value child must be a list",
               std::invalid_argument);

  auto val_device_view = column_device_view::create(val_view, stream);
  cudf::detail::lists_column_device_view val_lists_device_view(*val_device_view);

  // Initialize the null mask using the parent struct (or all-valid)
  auto null_mask =
    variant_column.nullable()
      ? cudf::detail::copy_bitmask(variant_column, stream, mr)
      : cudf::create_null_mask(variant_column.size(), mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  auto launch_int = [&]<typename T>(std::in_place_type_t<T>) {
    rmm::device_buffer data{static_cast<std::size_t>(num_rows) * sizeof(T), stream, mr};

    auto grid = cudf::detail::grid_1d{num_rows, block_size};
    cast_variant_int_kernel<T><<<grid.num_blocks, block_size, 0, stream.value()>>>(
      val_lists_device_view,
      {static_cast<T*>(data.data()), static_cast<std::size_t>(num_rows)},
      d_null_mask);

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
  cast_variant_string_fn fn{val_lists_device_view, d_null_mask, nullptr, nullptr, {}};
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
  auto intermediate = detail::get_variant_field(
    variant_column, path, stream, cudf::get_current_device_resource_ref());
  return detail::cast_variant(intermediate->view(), desired_type, stream, mr);
}

}  // namespace io::parquet::experimental
}  // namespace cudf
