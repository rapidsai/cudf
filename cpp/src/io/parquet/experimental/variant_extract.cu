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
#include <cudf/detail/utilities/batched_memcpy.hpp>
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
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace cudf {
namespace io::parquet::experimental {
namespace {

constexpr int variant_version_v1 = 1;

__device__ inline cuda::std::optional<uint64_t> read_uint(device_span<uint8_t const> data,
                                                          size_type pos,
                                                          int width)
{
  if (cuda::std::cmp_greater(pos + width, data.size())) { return cuda::std::nullopt; }
  uint64_t v = 0;
  memcpy(&v, data.data() + pos, width);
  return v;
}

// Safely narrow a decoded value to size_type
__device__ inline cuda::std::optional<size_type> narrow_cast(cuda::std::optional<uint64_t> value)
{
  if (!value.has_value() ||
      value.value() > static_cast<uint64_t>(cuda::std::numeric_limits<size_type>::max())) {
    return cuda::std::nullopt;
  }
  return static_cast<size_type>(value.value());
}

__device__ inline cuda::std::optional<uint64_t> variant_value_length(device_span<uint8_t const> enc)
{
  if (enc.size() < 1) { return cuda::std::nullopt; }
  uint8_t const vm       = enc[0];
  int const basic_type   = vm & 0x03;
  int const value_header = (vm >> 2) & 0x3F;

  if (basic_type == 0) {
    // Primitive: 1-byte header + a payload whose size is fixed by the physical type id, except
    // binary (15) and long string (16) which carry a 4-byte little-endian length prefix.
    switch (value_header) {
      case 0:                                // null
      case 1:                                // true
      case 2: return uint64_t{1};            // false
      case 3: return uint64_t{1 + 1};        // int8
      case 4: return uint64_t{1 + 2};        // int16
      case 5: return uint64_t{1 + 4};        // int32
      case 6: return uint64_t{1 + 8};        // int64
      case 7: return uint64_t{1 + 8};        // double
      case 8: return uint64_t{1 + 1 + 4};    // decimal4  (scale + int32)
      case 9: return uint64_t{1 + 1 + 8};    // decimal8  (scale + int64)
      case 10: return uint64_t{1 + 1 + 16};  // decimal16 (scale + int128)
      case 11: return uint64_t{1 + 4};       // date
      case 12: return uint64_t{1 + 8};       // timestamp micros
      case 13: return uint64_t{1 + 8};       // timestamp ntz micros
      case 14: return uint64_t{1 + 4};       // float
      case 15:                               // binary
      case 16: {                             // long string
        auto const len = read_uint(enc, 1, 4);
        if (!len.has_value()) { return cuda::std::nullopt; }
        return uint64_t{1 + 4} + len.value();
      }
      case 17: return uint64_t{1 + 8};   // time ntz micros
      case 18: return uint64_t{1 + 8};   // timestamp nanos
      case 19: return uint64_t{1 + 8};   // timestamp ntz nanos
      case 20: return uint64_t{1 + 16};  // uuid
      default: return cuda::std::nullopt;
    }
  }

  if (basic_type == 1) {
    // Short string: header encodes the length directly in value_header.
    return uint64_t{1} + static_cast<uint64_t>(value_header);
  }

  // Object (2) / array (3): the total payload size is the sentinel (last) offset entry; add the
  // bytes consumed by the header, num_elements, optional field-id list, and the offset list.
  bool const is_object     = basic_type == 2;
  int const field_off_size = (value_header & 0x03) + 1;
  int const field_id_size  = is_object ? (((value_header >> 2) & 0x03) + 1) : 0;
  bool const is_large =
    is_object ? (((value_header >> 4) & 0x01) != 0) : (((value_header >> 2) & 0x01) != 0);
  int const num_elts_size = is_large ? 4 : 1;

  auto const num_elts = read_uint(enc, 1, num_elts_size);
  if (!num_elts.has_value()) { return cuda::std::nullopt; }
  auto const n = num_elts.value();

  auto const offs_start  = uint64_t{1} + num_elts_size + n * field_id_size;
  auto const values_base = offs_start + (n + 1) * field_off_size;
  // Sentinel offset (entry n) holds the total size of the values region.
  auto const sentinel_pos = narrow_cast(offs_start + n * field_off_size);
  if (!sentinel_pos.has_value()) { return cuda::std::nullopt; }
  auto const sentinel = read_uint(enc, sentinel_pos.value(), field_off_size);
  if (!sentinel.has_value()) { return cuda::std::nullopt; }
  return values_base + sentinel.value();
}

__device__ inline cuda::std::optional<size_type> find_key_in_metadata(
  device_span<uint8_t const> meta, cudf::string_view key)
{
  auto const meta_len = static_cast<size_type>(meta.size());
  if (meta_len < 1) { return cuda::std::nullopt; }

  uint8_t const header = meta[0];
  int const version    = header & 0x0F;
  if (version != variant_version_v1) { return cuda::std::nullopt; }
  int const offset_size = ((header >> 6) & 0x03) + 1;

  size_type pos          = 1;
  auto const num_entries = narrow_cast(read_uint(meta, pos, offset_size));
  if (!num_entries.has_value()) { return cuda::std::nullopt; }
  pos += offset_size;

  size_type const offsets_start = pos;
  auto const offsets_bytes      = (static_cast<uint64_t>(num_entries.value()) + 1) * offset_size;
  if (offsets_bytes > static_cast<uint64_t>(meta_len - offsets_start)) {
    return cuda::std::nullopt;
  }

  auto start_off = read_uint(meta, offsets_start, offset_size);
  if (!start_off.has_value()) { return cuda::std::nullopt; }
  auto const strings_base = offsets_start + static_cast<size_type>(offsets_bytes);
  // Bytes available for dictionary string payloads
  auto const strings_extent = meta_len - strings_base;
  for (size_type i = 0; i < num_entries.value(); ++i) {
    auto const end_off = read_uint(meta, offsets_start + (i + 1) * offset_size, offset_size);
    if (!end_off.has_value()) { return cuda::std::nullopt; }
    if (end_off.value() < start_off.value() || end_off.value() > strings_extent) {
      return cuda::std::nullopt;
    }
    cudf::string_view const entry{
      reinterpret_cast<char const*>(meta.data() + strings_base + start_off.value()),
      static_cast<size_type>(end_off.value() - start_off.value())};
    if (entry == key) { return i; }
    start_off = end_off;
  }
  return cuda::std::nullopt;
}

__device__ inline device_span<uint8_t const> locate_object_field(device_span<uint8_t const> val,
                                                                 int field_id)
{
  auto const val_len = static_cast<size_type>(val.size());
  if (val_len < 1) { return {}; }
  uint8_t const vm     = val[0];
  int const basic_type = vm & 0x03;
  if (basic_type != 2) { return {}; }  // not object

  int const value_header   = (vm >> 2) & 0x3F;
  int const field_off_size = (value_header & 0x03) + 1;
  int const field_id_size  = ((value_header >> 2) & 0x03) + 1;
  int const num_elts_size  = ((value_header >> 4) & 0x01) != 0 ? 4 : 1;

  size_type pos         = 1;
  auto const num_fields = narrow_cast(read_uint(val, pos, num_elts_size));
  if (!num_fields.has_value()) { return {}; }
  pos += num_elts_size;

  size_type const field_ids_start = pos;
  auto const field_ids_bytes      = static_cast<uint64_t>(num_fields.value()) * field_id_size;
  if (field_ids_bytes > val_len - field_ids_start) { return {}; }

  size_type const field_offs_start = field_ids_start + static_cast<size_type>(field_ids_bytes);
  auto const field_offs_bytes = (static_cast<uint64_t>(num_fields.value()) + 1) * field_off_size;
  if (field_offs_bytes > val_len - field_offs_start) { return {}; }

  size_type const values_base = field_offs_start + static_cast<size_type>(field_offs_bytes);
  // Maximum legitimate field-offset value: bytes available after values_base
  auto const values_extent = val_len - values_base;

  // Find the matching field ID and its start offset
  bool found           = false;
  uint64_t match_start = 0;
  for (size_type i = 0; i < num_fields.value(); ++i) {
    auto const fid = read_uint(val, field_ids_start + i * field_id_size, field_id_size);
    if (!fid.has_value()) { return {}; }
    if (cuda::std::cmp_not_equal(fid.value(), field_id)) { continue; }

    auto const o = read_uint(val, field_offs_start + i * field_off_size, field_off_size);
    if (!o.has_value()) { return {}; }
    if (o.value() > values_extent) { return {}; }
    match_start = o.value();
    found       = true;
    break;
  }
  if (!found) { return {}; }

  // Derive field's value length from its header
  auto const field_value = val.subspan(values_base + match_start);
  auto const value_len   = variant_value_length(field_value);
  if (!value_len.has_value()) { return {}; }
  auto const match_end = match_start + value_len.value();
  if (match_end > values_extent) { return {}; }
  return val.subspan(values_base + match_start, value_len.value());
}

// Variant primitive ints: basic_type=0, value_header maps INT{8,16,32,64} -> {3,4,5,6}.
template <typename T>
__device__ inline cuda::std::optional<T> decode_int(device_span<uint8_t const> enc)
{
  static_assert(cuda::std::is_same_v<T, int8_t> || cuda::std::is_same_v<T, int16_t> ||
                  cuda::std::is_same_v<T, int32_t> || cuda::std::is_same_v<T, int64_t>,
                "decode_int: T must be int8_t, int16_t, int32_t, or int64_t");
  constexpr int expected_value_header = cuda::std::is_same_v<T, int8_t>    ? 3
                                        : cuda::std::is_same_v<T, int16_t> ? 4
                                        : cuda::std::is_same_v<T, int32_t> ? 5
                                                                           : 6;
  constexpr int width                 = sizeof(T);
  if (cuda::std::cmp_less(enc.size(), 1 + width)) { return cuda::std::nullopt; }
  uint8_t const vm = enc[0];
  if ((vm & 0x03) != 0 || ((vm >> 2) & 0x3F) != expected_value_header) {
    return cuda::std::nullopt;
  }
  T v;
  memcpy(&v, enc.data() + 1, width);
  return v;
}

// Walk a path of object-key steps level by level starting at `val` and return the span of the
// final value (subspan of `val`). Returns an empty span on failure.
__device__ inline device_span<uint8_t const> locate_path(device_span<uint8_t const> meta,
                                                         device_span<uint8_t const> val,
                                                         column_device_view path)
{
  device_span<uint8_t const> sub_val = val;
  for (size_type i = 0; i < path.size(); ++i) {
    auto const name = path.element<cudf::string_view>(i);

    auto const field_id = find_key_in_metadata(meta, name);
    if (!field_id.has_value()) { return {}; }

    sub_val = locate_object_field(sub_val, field_id.value());
    if (sub_val.empty()) { return {}; }
  }
  return sub_val;
}

__device__ inline cuda::std::optional<device_span<uint8_t const>> decode_string(
  device_span<uint8_t const> enc)
{
  auto const len = enc.size();
  if (len < 1) { return cuda::std::nullopt; }
  uint8_t const vm       = enc[0];
  int const basic_type   = vm & 0x03;
  int const value_header = (vm >> 2) & 0x3F;

  if (basic_type == 1) {
    // Short string: value_header = length
    std::size_t const str_len = value_header;
    if (1 + str_len > len) { return cuda::std::nullopt; }
    return enc.subspan(1, str_len);
  }
  if (basic_type == 0 && value_header == 16) {
    // Long string: 1-byte header + 4-byte LE length + char bytes
    constexpr std::size_t long_string_prefix_bytes = 1 + sizeof(uint32_t);
    if (len < long_string_prefix_bytes) { return cuda::std::nullopt; }
    uint32_t str_len;
    memcpy(&str_len, enc.data() + 1, sizeof(str_len));
    // Encoded length claims more char bytes than the buffer holds: truncated/malformed blob
    if (long_string_prefix_bytes + str_len > len) { return cuda::std::nullopt; }
    return enc.subspan(long_string_prefix_bytes, str_len);
  }
  return cuda::std::nullopt;
}

// Returns the metadata and value list bytes for a given row from device views
__device__ inline cuda::std::pair<device_span<uint8_t const>, device_span<uint8_t const>>
metadata_and_value_at(cudf::detail::lists_column_device_view const& metadata,
                      cudf::detail::lists_column_device_view const& values,
                      size_type row)
{
  auto const meta_begin = metadata.offset_at(row);
  auto const meta_end   = metadata.offset_at(row + 1);
  auto const val_begin  = values.offset_at(row);
  auto const val_end    = values.offset_at(row + 1);
  return {
    {metadata.child().data<uint8_t>() + meta_begin,
     static_cast<std::size_t>(meta_end - meta_begin)},
    {values.child().data<uint8_t>() + val_begin, static_cast<std::size_t>(val_end - val_begin)}};
}

constexpr int block_size = 256;

CUDF_KERNEL __launch_bounds__(block_size) void locate_variant_fields_kernel(
  cudf::detail::lists_column_device_view metadata,
  cudf::detail::lists_column_device_view values,
  column_device_view path,
  device_span<size_type> d_sizes,
  device_span<size_type> d_src_offsets,
  bitmask_type* d_null_mask)
{
  auto const num_rows = static_cast<size_type>(d_sizes.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }
  auto const row = tid;

  if (!cudf::bit_is_set(d_null_mask, row)) {
    d_sizes[row]       = 0;
    d_src_offsets[row] = 0;
    return;
  }

  auto const [meta, val] = metadata_and_value_at(metadata, values, row);

  auto const field = locate_path(meta, val, path);
  if (field.empty()) {
    d_sizes[row]       = 0;
    d_src_offsets[row] = 0;
    cudf::clear_bit(d_null_mask, row);
    return;
  }

  d_sizes[row]       = static_cast<size_type>(field.size());
  d_src_offsets[row] = static_cast<size_type>(field.data() - val.data());
}

template <typename T>
CUDF_KERNEL __launch_bounds__(block_size) void cast_variant_int_kernel(
  cudf::detail::lists_column_device_view values, device_span<T> d_output, bitmask_type* d_null_mask)
{
  auto const num_rows = static_cast<size_type>(d_output.size());
  auto const tid      = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }
  auto const row = tid;

  if (!cudf::bit_is_set(d_null_mask, row)) {
    d_output[row] = 0;
    return;
  }

  auto const val_begin = values.offset_at(row);
  auto const val_end   = values.offset_at(row + 1);
  auto const val_child = values.child();
  device_span<uint8_t const> const val{val_child.data<uint8_t>() + val_begin,
                                       static_cast<std::size_t>(val_end - val_begin)};

  auto const decoded = decode_int<T>(val);
  if (decoded.has_value()) {
    d_output[row] = *decoded;
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
    device_span<uint8_t const> const val{val_child.data<uint8_t>() + val_begin,
                                         static_cast<std::size_t>(val_end - val_begin)};

    auto const str = decode_string(val);
    if (!str) {
      if (!d_chars) { d_sizes[row] = 0; }
      cudf::clear_bit(d_null_mask, row);
      return;
    }

    if (!d_chars) {
      d_sizes[row] = str->size();
    } else {
      memcpy(d_chars + d_offsets[row], str->data(), str->size());
    }
  }
};

void validate_variant_child(column_view const& child)
{
  CUDF_EXPECTS(child.type().id() == type_id::LIST,
               "VARIANT metadata/value column must be a list",
               std::invalid_argument);
  CUDF_EXPECTS(lists_column_view{child}.child().type().id() == type_id::UINT8,
               "VARIANT metadata/value column must be list<uint8>",
               std::invalid_argument);
}

struct cast_variant_launcher {
  cudf::detail::lists_column_device_view values;
  size_type num_rows;
  data_type desired_type;
  bitmask_type* d_null_mask;
  rmm::device_buffer null_mask;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(cuda::std::is_same_v<T, int8_t> || cuda::std::is_same_v<T, int16_t> ||
             cuda::std::is_same_v<T, int32_t> || cuda::std::is_same_v<T, int64_t>)
  {
    rmm::device_buffer data{num_rows * sizeof(T), stream, mr};

    auto grid = cudf::detail::grid_1d{num_rows, block_size};
    cast_variant_int_kernel<T><<<grid.num_blocks, block_size, 0, stream.value()>>>(
      values, {static_cast<T*>(data.data()), static_cast<std::size_t>(num_rows)}, d_null_mask);

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return std::make_unique<column>(desired_type,
                                    num_rows,
                                    std::move(data),
                                    null_count > 0 ? std::move(null_mask) : rmm::device_buffer{},
                                    null_count);
  }

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(cuda::std::is_same_v<T, cudf::string_view>)
  {
    cast_variant_string_fn fn{values, d_null_mask, nullptr, nullptr, {}};
    auto [offsets_column, chars] =
      cudf::strings::detail::make_strings_children(fn, num_rows, stream, mr);

    auto const null_count =
      num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
    return make_strings_column(num_rows,
                               std::move(offsets_column),
                               chars.release(),
                               null_count,
                               null_count > 0 ? std::move(null_mask) : rmm::device_buffer{});
  }

  template <typename T>
  std::unique_ptr<column> operator()()
    requires(not(cuda::std::is_same_v<T, int8_t> || cuda::std::is_same_v<T, int16_t> ||
                 cuda::std::is_same_v<T, int32_t> || cuda::std::is_same_v<T, int64_t> ||
                 cuda::std::is_same_v<T, cudf::string_view>))
  {
    CUDF_FAIL("unsupported type for variant cast", std::invalid_argument);
  }
};

std::unique_ptr<column> build_path_column(std::vector<std::string> const& steps,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const depth = steps.size();

  std::string host_chars;
  std::vector<size_type> host_offsets(depth + 1);
  for (size_t i = 0; i < depth; ++i) {
    host_offsets[i] = static_cast<size_type>(host_chars.size());
    host_chars.append(steps[i]);
  }
  host_offsets[depth] = host_chars.size();

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
  // Validate the variant column
  CUDF_EXPECTS(variant_column.type().id() == type_id::STRUCT,
               "VARIANT column must be struct type",
               std::invalid_argument);
  CUDF_EXPECTS(variant_column.num_children() >= 2,
               "VARIANT struct must have at least two children",
               std::invalid_argument);
  validate_variant_child(variant_column.child(0));
  validate_variant_child(variant_column.child(1));

  // Validate the path even for empty input columns
  auto const steps = parse_variant_path(path);

  auto const num_rows = variant_column.size();
  if (num_rows == 0) {
    return cudf::make_lists_column(
      0, make_empty_column(type_id::INT32), make_empty_column(type_id::UINT8), 0, {});
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();

  auto path_column      = build_path_column(steps, stream, temp_mr);
  auto path_device_view = column_device_view::create(path_column->view(), stream);

  // Resolve children with respect to any slice/offset on the parent struct
  structs_column_view const variant_struct{variant_column};
  auto const meta_view = variant_struct.get_sliced_child(0, stream);
  auto const val_view  = variant_struct.get_sliced_child(1, stream);

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

  // Parse the path per row and compute the output sizes
  auto grid = cudf::detail::grid_1d{num_rows, block_size};
  locate_variant_fields_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
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

  // Copy values into the output buffer
  auto val_child = make_numeric_column(
    data_type{type_id::UINT8}, total_bytes, mask_state::UNALLOCATED, stream, mr);
  if (total_bytes > 0) {
    auto const out_base = val_child->mutable_view().data<uint8_t>();
    auto src_iter       = thrust::make_transform_iterator(
      thrust::counting_iterator<size_type>(0),
      cuda::proclaim_return_type<uint8_t const*>(
        [vlv   = val_lists_device_view,
         d_src = d_src_offsets.data()] __device__(size_type row) -> uint8_t const* {
          return vlv.child().template data<uint8_t>() + vlv.offset_at(row) + d_src[row];
        }));
    auto dst_iter = thrust::make_transform_iterator(
      thrust::counting_iterator<size_type>(0),
      cuda::proclaim_return_type<uint8_t*>(
        [out_base, d_off = d_offsets.data()] __device__(size_type row) -> uint8_t* {
          return out_base + d_off[row];
        }));
    cudf::detail::batched_memcpy_async(src_iter, dst_iter, d_sizes.begin(), num_rows, stream);
  }

  auto const null_count = num_rows - cudf::detail::count_set_bits(d_null_mask, 0, num_rows, stream);
  return make_lists_column(num_rows,
                           std::move(offsets_column),
                           std::move(val_child),
                           null_count,
                           null_count > 0 ? std::move(null_mask) : rmm::device_buffer{});
}

std::unique_ptr<column> cast_variant(column_view const& values,
                                     data_type desired_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  validate_variant_child(values);
  size_type const num_rows = values.size();

  auto val_device_view = column_device_view::create(values, stream);
  cudf::detail::lists_column_device_view val_lists_device_view(*val_device_view);

  // Initialize the null mask from the values column (or all-valid)
  auto null_mask    = values.nullable()
                        ? cudf::detail::copy_bitmask(values, stream, mr)
                        : cudf::create_null_mask(num_rows, mask_state::ALL_VALID, stream, mr);
  auto* d_null_mask = static_cast<bitmask_type*>(null_mask.data());

  return cudf::type_dispatcher(desired_type,
                               cast_variant_launcher{val_lists_device_view,
                                                     num_rows,
                                                     desired_type,
                                                     d_null_mask,
                                                     std::move(null_mask),
                                                     stream,
                                                     mr});
}

}  // namespace detail

std::unique_ptr<column> get_variant_field(column_view const& variant_column,
                                          std::string_view path,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_variant_field(variant_column, path, stream, mr);
}

std::unique_ptr<column> cast_variant(column_view const& values,
                                     data_type desired_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::cast_variant(values, desired_type, stream, mr);
}

std::unique_ptr<column> extract_variant_field(column_view const& variant_column,
                                              std::string_view path,
                                              data_type desired_type,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto value = detail::get_variant_field(
    variant_column, path, stream, cudf::get_current_device_resource_ref());
  return detail::cast_variant(value->view(), desired_type, stream, mr);
}

}  // namespace io::parquet::experimental
}  // namespace cudf
