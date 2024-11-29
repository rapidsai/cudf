/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file write_json.cu
 * @brief cuDF-IO JSON writer implementation
 */

#include "io/comp/comp.hpp"
#include "io/csv/durations.hpp"
#include "io/utilities/parsing_utils.cuh"
#include "lists/utilities.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace cudf::io::json::detail {

std::unique_ptr<column> make_column_names_column(host_span<column_name_info const> column_names,
                                                 size_type num_columns,
                                                 rmm::cuda_stream_view stream);
namespace {

/**
 * @brief Functor to modify a string column for JSON format.
 *
 * This will convert escape characters and wrap quotes around strings.
 */
struct escape_strings_fn {
  column_device_view const d_column;
  bool const append_colon{false};
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void write_char(char_utf8 chr, char*& d_buffer, size_type& bytes)
  {
    if (d_buffer)
      d_buffer += cudf::strings::detail::from_char_utf8(chr, d_buffer);
    else
      bytes += cudf::strings::detail::bytes_in_char_utf8(chr);
  }

  __device__ inline char nibble_to_hex(uint8_t nibble) const
  {
    return nibble < 10 ? '0' + nibble : 'a' + nibble - 10;
  }

  __device__ void write_utf8_codepoint(uint16_t codepoint, char*& d_buffer, size_type& bytes)
  {
    if (d_buffer) {
      d_buffer[0] = '\\';
      d_buffer[1] = 'u';
      d_buffer[2] = nibble_to_hex((codepoint >> 12) & 0x0F);
      d_buffer[3] = nibble_to_hex((codepoint >> 8) & 0x0F);
      d_buffer[4] = nibble_to_hex((codepoint >> 4) & 0x0F);
      d_buffer[5] = nibble_to_hex((codepoint)&0x0F);
      d_buffer += 6;
    } else {
      bytes += 6;
    }
  }

  __device__ void write_utf16_codepoint(uint32_t codepoint, char*& d_buffer, size_type& bytes)
  {
    constexpr uint16_t UTF16_HIGH_SURROGATE_BEGIN = 0xD800;
    constexpr uint16_t UTF16_LOW_SURROGATE_BEGIN  = 0xDC00;
    codepoint -= 0x1'0000;
    uint16_t hex_high = ((codepoint >> 10) & 0x3FF) + UTF16_HIGH_SURROGATE_BEGIN;
    uint16_t hex_low  = (codepoint & 0x3FF) + UTF16_LOW_SURROGATE_BEGIN;
    write_utf8_codepoint(hex_high, d_buffer, bytes);
    write_utf8_codepoint(hex_low, d_buffer, bytes);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    auto const d_str = d_column.element<string_view>(idx);

    // entire string must be double-quoted.
    constexpr char_utf8 const quote = '\"';  // wrap quotes
    bool constexpr quote_row        = true;

    char* d_buffer  = d_chars ? d_chars + d_offsets[idx] : nullptr;
    size_type bytes = 0;

    if (quote_row) write_char(quote, d_buffer, bytes);
    for (auto utf8_char : d_str) {
      if (utf8_char > 0x0000'00FF) {
        // multi-byte char
        uint32_t codepoint = cudf::strings::detail::utf8_to_codepoint(utf8_char);
        if (codepoint <= 0x0000'FFFF) {
          // write \uXXXX utf-8 codepoint
          write_utf8_codepoint(codepoint, d_buffer, bytes);
        } else {
          // write \uXXXX\uXXXX utf-16 surrogate pair
          // codepoint > 0xFFFF && codepoint <= 0x10FFFF
          write_utf16_codepoint(codepoint, d_buffer, bytes);
        }
        continue;
      }
      auto escaped_chars = get_escaped_char(utf8_char);
      if (escaped_chars.first == '\0') {
        write_char(escaped_chars.second, d_buffer, bytes);
      } else {
        write_char(escaped_chars.first, d_buffer, bytes);
        write_char(escaped_chars.second, d_buffer, bytes);
      }
    }
    if (quote_row) write_char(quote, d_buffer, bytes);
    constexpr char_utf8 const colon = ':';  // append colon
    if (append_colon) write_char(colon, d_buffer, bytes);

    if (!d_chars) { d_sizes[idx] = bytes; }
  }

  std::unique_ptr<column> get_escaped_strings(column_view const& column_v,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
  {
    if (column_v.is_empty()) {  // empty begets empty
      return make_empty_column(type_id::STRING);
    }
    auto [offsets_column, chars] =
      cudf::strings::detail::make_strings_children(*this, column_v.size(), stream, mr);

    return make_strings_column(column_v.size(),
                               std::move(offsets_column),
                               chars.release(),
                               column_v.null_count(),
                               cudf::detail::copy_bitmask(column_v, stream, mr));
  }
};

// Struct - scatter string_views of each element in a struct column
struct struct_scatter_strings_fn {
  table_device_view const tbl;
  column_device_view const col_names;
  size_type const strviews_per_column;
  size_type const num_strviews_per_row;
  string_view const row_prefix;       // "{"
  string_view const row_suffix;       // "}" or "}\n" for json-lines
  string_view const value_separator;  // ","
  string_view const narep;            // null entry replacement
  bool const include_nulls;
  string_view* d_strviews;

  /**
   * @brief Scatters string_views for each element in a struct column
   *
   * @param idx Column-major index of the element to scatter
   */
  __device__ void operator()(size_type idx)
  {
    auto const row        = idx / tbl.num_columns();
    auto const col        = idx % tbl.num_columns();
    auto const d_str_null = tbl.column(col).is_null(row);
    auto const this_index = row * num_strviews_per_row + col * strviews_per_column + 1;
    // prefix
    if (col == 0) d_strviews[this_index - 1] = row_prefix;
    if (col != 0) d_strviews[this_index - 1] = include_nulls ? value_separator : string_view{};
    if (!include_nulls && d_str_null) {
      d_strviews[this_index]     = string_view{};
      d_strviews[this_index + 1] = string_view{};
    } else {
      auto const d_col_name = col_names.element<string_view>(col);
      auto const d_str = d_str_null ? narep : tbl.column(col).template element<string_view>(row);
      // column_name: value
      d_strviews[this_index]     = d_col_name;
      d_strviews[this_index + 1] = d_str;
    }
    // suffix
    if (col == tbl.num_columns() - 1) { d_strviews[this_index + 2] = row_suffix; }
  }
};

struct validity_fn {
  table_device_view const tbl;
  __device__ bool operator()(size_type idx) const
  {
    auto const row = idx / tbl.num_columns();
    auto const col = idx % tbl.num_columns();
    return tbl.column(col).is_valid(row);
  }
};

/**
 * @brief Concatenate the strings from each row of the given table as structs in JSON string
 *
 * Each row will be struct with field name as column names and values from each column in the table.
 *
 * @param strings_columns Table of strings columns
 * @param column_names Column of names for each column in the table
 * @param num_rows Number of rows in the table
 * @param row_prefix Prepend this string to each row
 * @param row_suffix  Append this string to each row
 * @param value_separator Separator between values
 * @param narep Null-String replacement
 * @param include_nulls Include null string entries in the output
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource to use for device memory allocation.
 * @return New strings column of JSON structs in each row
 */
std::unique_ptr<column> struct_to_strings(table_view const& strings_columns,
                                          column_view const& column_names,
                                          size_type const num_rows,
                                          string_view const row_prefix,
                                          string_view const row_suffix,
                                          string_view const value_separator,
                                          string_scalar const& narep,
                                          bool include_nulls,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(column_names.type().id() == type_id::STRING, "Column names must be of type string");
  auto const num_columns = strings_columns.num_columns();
  CUDF_EXPECTS(num_columns == column_names.size(),
               "Number of column names should be equal to number of columns in the table");
  if (num_rows == 0)  // empty begets empty
    return make_empty_column(type_id::STRING);
  // check all columns are of type string
  CUDF_EXPECTS(std::all_of(strings_columns.begin(),
                           strings_columns.end(),
                           [](auto const& c) { return c.type().id() == type_id::STRING; }),
               "All columns must be of type string");
  auto constexpr strviews_per_column = 3;  // (for each "column_name:", "value",  "separator")
  auto const num_strviews_per_row    = strings_columns.num_columns() == 0
                                         ? 2
                                         : (1 + strings_columns.num_columns() * strviews_per_column);
  // e.g. {col1: value, col2: value, col3: value} = 1 + 3 + 3 + (3-1) + 1 = 10

  auto tbl_device_view = cudf::table_device_view::create(strings_columns, stream);
  auto d_column_names  = column_device_view::create(column_names, stream);

  // Note for future: chunk it but maximize parallelism, if memory usage is high.
  auto const total_strings = num_strviews_per_row * num_rows;
  auto const total_rows    = num_rows * strings_columns.num_columns();
  rmm::device_uvector<string_view> d_strviews(total_strings, stream);
  if (strings_columns.num_columns() > 0) {
    struct_scatter_strings_fn scatter_fn{*tbl_device_view,
                                         *d_column_names,
                                         strviews_per_column,
                                         num_strviews_per_row,
                                         row_prefix,
                                         row_suffix,
                                         value_separator,
                                         narep.value(stream),
                                         include_nulls,
                                         d_strviews.begin()};
    // scatter row_prefix, row_suffix, column_name:, value, value_separator as string_views
    thrust::for_each(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     thrust::make_counting_iterator<size_type>(total_rows),
                     scatter_fn);
  } else {
    thrust::for_each(
      rmm::exec_policy_nosync(stream),
      thrust::make_counting_iterator<size_type>(0),
      thrust::make_counting_iterator<size_type>(num_rows),
      [d_strviews = d_strviews.begin(), row_prefix, row_suffix, num_strviews_per_row] __device__(
        auto idx) {
        auto const this_index                             = idx * num_strviews_per_row;
        d_strviews[this_index]                            = row_prefix;
        d_strviews[this_index + num_strviews_per_row - 1] = row_suffix;
      });
  }
  if (!include_nulls) {
    // if previous column was null, then we skip the value separator
    rmm::device_uvector<bool> d_str_separator(total_rows, stream);
    auto row_num = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<size_type>([tbl = *tbl_device_view] __device__(auto idx)
                                              -> size_type { return idx / tbl.num_columns(); }));
    auto validity_iterator =
      cudf::detail::make_counting_transform_iterator(0, validity_fn{*tbl_device_view});
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream),
                                  row_num,
                                  row_num + total_rows,
                                  validity_iterator,
                                  d_str_separator.begin(),
                                  false,
                                  thrust::equal_to<size_type>{},
                                  thrust::logical_or<bool>{});
    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     thrust::make_counting_iterator<size_type>(total_rows),
                     [write_separator = d_str_separator.begin(),
                      d_strviews      = d_strviews.begin(),
                      value_separator,
                      tbl = *tbl_device_view,
                      strviews_per_column,
                      num_strviews_per_row] __device__(auto idx) {
                       auto const row = idx / tbl.num_columns();
                       auto const col = idx % tbl.num_columns();
                       auto const this_index =
                         row * num_strviews_per_row + col * strviews_per_column + 1;
                       if (write_separator[idx] && tbl.column(col).is_valid(row)) {
                         d_strviews[this_index - 1] = value_separator;
                       }
                     });
  }
  auto joined_col = make_strings_column(d_strviews, string_view{nullptr, 0}, stream, mr);

  // gather from offset and create a new string column
  auto old_offsets = strings_column_view(joined_col->view()).offsets();
  rmm::device_uvector<size_type> row_string_offsets(num_rows + 1, stream, mr);
  auto const d_strview_offsets = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_type>([num_strviews_per_row] __device__(size_type const i) {
      return i * num_strviews_per_row;
    }));
  thrust::gather(rmm::exec_policy(stream),
                 d_strview_offsets,
                 d_strview_offsets + row_string_offsets.size(),
                 old_offsets.begin<size_type>(),
                 row_string_offsets.begin());
  auto chars_data = joined_col->release().data;
  return make_strings_column(
    num_rows,
    std::make_unique<cudf::column>(std::move(row_string_offsets), rmm::device_buffer{}, 0),
    std::move(chars_data.release()[0]),
    0,
    {});
}

/**
 * @brief Concatenates a list of strings columns into a single strings column.
 *
 * @param lists_strings Column containing lists of strings to concatenate.
 * @param list_prefix String to place before each list. (typically [)
 * @param list_suffix String to place after each list. (typically ])
 * @param element_separator String that should inserted between strings of each list row.
 * @param element_narep String that should be used in place of any null strings.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column with concatenated results.
 */
std::unique_ptr<column> join_list_of_strings(lists_column_view const& lists_strings,
                                             string_view const list_prefix,
                                             string_view const list_suffix,
                                             string_view const element_separator,
                                             string_view const element_narep,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  /*
  create string_views of the list elements, and the list separators and list prefix/suffix.
  then concatenates them all together.
  gather offset of first string_view of each row as offsets for output string column.
  Algorithm:
    calculate #strviews per list using null mask, and list_offsets.
    scan #strviews to get strviews_offset
    create label segments.
    sublist_index = index - offsets[label]
    strviews_offset[label] + sublist_index = string_view index +1, +2
    use above 2 to scatter element, element_seperator
    scatter list_prefix, list_suffix to the right place using list_offsets
    make_strings_column() and gather offsets, based on strviews_offset.
  */
  auto const offsets          = lists_strings.offsets();
  auto const strings_children = lists_strings.get_sliced_child(stream);
  auto const num_lists        = lists_strings.size();
  auto const num_strings      = strings_children.size();
  auto const num_offsets      = offsets.size();

  rmm::device_uvector<size_type> d_strview_offsets(num_offsets, stream);
  auto num_strings_per_list = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>(
      [offsets = offsets.begin<size_type>(), num_offsets] __device__(size_type idx) {
        if (idx + 1 >= num_offsets) return 0;
        auto const length = offsets[idx + 1] - offsets[idx];
        return length == 0 ? 2 : (2 + length + length - 1);
      }));
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         num_strings_per_list,
                         num_strings_per_list + num_offsets,
                         d_strview_offsets.begin());
  auto const total_strings = d_strview_offsets.back_element(stream);

  rmm::device_uvector<string_view> d_strviews(total_strings, stream);
  // scatter null_list and list_prefix, list_suffix
  auto col_device_view = cudf::column_device_view::create(lists_strings.parent(), stream);
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator<size_type>(0),
                   thrust::make_counting_iterator<size_type>(num_lists),
                   [col = *col_device_view,
                    list_prefix,
                    list_suffix,
                    d_strview_offsets = d_strview_offsets.begin(),
                    d_strviews        = d_strviews.begin()] __device__(auto idx) {
                     if (col.is_null(idx)) {
                       d_strviews[d_strview_offsets[idx]]     = string_view{};
                       d_strviews[d_strview_offsets[idx] + 1] = string_view{};
                     } else {
                       // [ ]
                       d_strviews[d_strview_offsets[idx]]         = list_prefix;
                       d_strviews[d_strview_offsets[idx + 1] - 1] = list_suffix;
                     }
                   });

  // scatter string and separator
  auto labels = cudf::lists::detail::generate_labels(
    lists_strings, num_strings, stream, cudf::get_current_device_resource_ref());
  auto d_strings_children = cudf::column_device_view::create(strings_children, stream);
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator<size_type>(0),
                   thrust::make_counting_iterator<size_type>(num_strings),
                   [col                = *col_device_view,
                    d_strview_offsets  = d_strview_offsets.begin(),
                    d_strviews         = d_strviews.begin(),
                    labels             = labels->view().begin<size_type>(),
                    list_offsets       = offsets.begin<size_type>(),
                    d_strings_children = *d_strings_children,
                    element_separator,
                    element_narep] __device__(auto idx) {
                     auto const label         = labels[idx];
                     auto const sublist_index = idx - list_offsets[label];
                     auto const strview_index = d_strview_offsets[label] + sublist_index * 2 + 1;
                     // value or na_rep
                     auto const strview = d_strings_children.element<cudf::string_view>(idx);
                     d_strviews[strview_index] =
                       d_strings_children.is_null(idx) ? element_narep : strview;
                     // separator
                     if (sublist_index != 0) { d_strviews[strview_index - 1] = element_separator; }
                   });

  auto joined_col = make_strings_column(d_strviews, string_view{nullptr, 0}, stream, mr);

  // gather from offset and create a new string column
  auto old_offsets = strings_column_view(joined_col->view()).offsets();
  rmm::device_uvector<size_type> row_string_offsets(num_offsets, stream, mr);
  thrust::gather(rmm::exec_policy(stream),
                 d_strview_offsets.begin(),
                 d_strview_offsets.end(),
                 old_offsets.begin<size_type>(),
                 row_string_offsets.begin());
  auto chars_data = joined_col->release().data;
  return make_strings_column(
    num_lists,
    std::make_unique<cudf::column>(std::move(row_string_offsets), rmm::device_buffer{}, 0),
    std::move(chars_data.release()[0]),
    lists_strings.null_count(),
    cudf::detail::copy_bitmask(lists_strings.parent(), stream, mr));
}

/**
 * @brief Functor to convert a column to string representation for JSON format.
 */
struct column_to_strings_fn {
  /**
   * @brief Returns true if the specified type is not supported by the JSON writer.
   */
  template <typename column_type>
  constexpr static bool is_not_handled()
  {
    // Note: the case (not std::is_same_v<column_type, bool>)  is already covered by is_integral)
    return not((std::is_same_v<column_type, cudf::string_view>) ||
               (std::is_integral_v<column_type>) || (std::is_floating_point_v<column_type>) ||
               (cudf::is_fixed_point<column_type>()) || (cudf::is_timestamp<column_type>()) ||
               (cudf::is_duration<column_type>()));
  }

  explicit column_to_strings_fn(json_writer_options const& options,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
    : options_(options),
      stream_(stream),
      mr_(mr),
      narep(options.get_na_rep(), true, stream),
      struct_value_separator(",", true, stream),
      struct_row_begin_wrap("{", true, stream),
      struct_row_end_wrap("}", true, stream),
      list_value_separator(",", true, stream),
      list_row_begin_wrap("[", true, stream),
      list_row_end_wrap("]", true, stream),
      true_value(options_.get_true_value(), true, stream),
      false_value(options_.get_false_value(), true, stream)
  {
  }

  ~column_to_strings_fn()                                      = default;
  column_to_strings_fn(column_to_strings_fn const&)            = delete;
  column_to_strings_fn& operator=(column_to_strings_fn const&) = delete;
  column_to_strings_fn(column_to_strings_fn&&)                 = delete;
  column_to_strings_fn& operator=(column_to_strings_fn&&)      = delete;

  // unsupported type of column:
  template <typename column_type>
  std::enable_if_t<is_not_handled<column_type>(), std::unique_ptr<column>> operator()(
    column_view const&) const
  {
    CUDF_FAIL("Unsupported column type.");
  }

  // Note: `null` replacement with `na_rep` deferred to `concatenate()`
  // instead of column-wise; might be faster.

  // bools:
  template <typename column_type>
  std::enable_if_t<std::is_same_v<column_type, bool>, std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    return cudf::strings::detail::from_booleans(column, true_value, false_value, stream_, mr_);
  }

  // strings:
  template <typename column_type>
  std::enable_if_t<std::is_same_v<column_type, cudf::string_view>, std::unique_ptr<column>>
  operator()(column_view const& column_v) const
  {
    auto d_column = column_device_view::create(column_v, stream_);
    return escape_strings_fn{*d_column}.get_escaped_strings(column_v, stream_, mr_);
  }

  // ints:
  template <typename column_type>
  std::enable_if_t<std::is_integral_v<column_type> && !std::is_same_v<column_type, bool>,
                   std::unique_ptr<column>>
  operator()(column_view const& column) const
  {
    return cudf::strings::detail::from_integers(column, stream_, mr_);
  }

  // floats:
  template <typename column_type>
  std::enable_if_t<std::is_floating_point_v<column_type>, std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    return cudf::strings::detail::from_floats(column, stream_, mr_);
  }

  // fixed point:
  template <typename column_type>
  std::enable_if_t<cudf::is_fixed_point<column_type>(), std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    return cudf::strings::detail::from_fixed_point(column, stream_, mr_);
  }

  // timestamps:
  template <typename column_type>
  std::enable_if_t<cudf::is_timestamp<column_type>(), std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    std::string format = [&]() {
      if (std::is_same_v<cudf::timestamp_s, column_type>) {
        return std::string{"%Y-%m-%dT%H:%M:%SZ"};
      } else if (std::is_same_v<cudf::timestamp_ms, column_type>) {
        return std::string{"%Y-%m-%dT%H:%M:%S.%3fZ"};
      } else if (std::is_same_v<cudf::timestamp_us, column_type>) {
        return std::string{"%Y-%m-%dT%H:%M:%S.%6fZ"};
      } else if (std::is_same_v<cudf::timestamp_ns, column_type>) {
        return std::string{"%Y-%m-%dT%H:%M:%S.%9fZ"};
      } else {
        return std::string{"%Y-%m-%d"};
      }
    }();

    // Since format uses ":", we need to add quotes to the format
    format = "\"" + format + "\"";

    return cudf::strings::detail::from_timestamps(
      column,
      format,
      strings_column_view(make_empty_column(type_id::STRING)->view()),
      stream_,
      mr_);
  }

  template <typename column_type>
  std::enable_if_t<cudf::is_duration<column_type>(), std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    auto duration_string = cudf::io::detail::csv::pandas_format_durations(column, stream_, mr_);
    auto quotes =
      make_column_from_scalar(string_scalar{"\"", true, stream_}, column.size(), stream_, mr_);
    return cudf::strings::detail::concatenate(
      table_view{{quotes->view(), duration_string->view(), quotes->view()}},
      string_scalar("", true, stream_),
      string_scalar("", false, stream_),
      strings::separator_on_nulls::YES,
      stream_,
      mr_);
  }

  // lists:
  template <typename column_type>
  std::enable_if_t<std::is_same_v<column_type, cudf::list_view>, std::unique_ptr<column>>
  operator()(column_view const& column, host_span<column_name_info const> children_names) const
  {
    auto child_view            = lists_column_view(column).get_sliced_child(stream_);
    auto constexpr child_index = lists_column_view::child_column_index;

    auto child_string_with_null = [&]() {
      if (child_view.type().id() == type_id::STRUCT) {
        return this->template operator()<cudf::struct_view>(child_view,
                                                            children_names.size() > child_index
                                                              ? children_names[child_index].children
                                                              : std::vector<column_name_info>{});
      } else if (child_view.type().id() == type_id::LIST) {
        return this->template operator()<cudf::list_view>(child_view,
                                                          children_names.size() > child_index
                                                            ? children_names[child_index].children
                                                            : std::vector<column_name_info>{});
      } else {
        return cudf::type_dispatcher<cudf::id_to_type_impl, column_to_strings_fn const&>(
          child_view.type(), *this, child_view);
      }
    };
    auto new_offsets = cudf::lists::detail::get_normalized_offsets(
      lists_column_view(column), stream_, cudf::get_current_device_resource_ref());
    auto const list_child_string = make_lists_column(
      column.size(),
      std::move(new_offsets),
      child_string_with_null(),
      column.null_count(),
      cudf::detail::copy_bitmask(column, stream_, cudf::get_current_device_resource_ref()),
      stream_);
    return join_list_of_strings(lists_column_view(*list_child_string),
                                list_row_begin_wrap.value(stream_),
                                list_row_end_wrap.value(stream_),
                                list_value_separator.value(stream_),
                                narep.value(stream_),
                                stream_,
                                mr_);
  }

  // structs:
  template <typename column_type>
  std::enable_if_t<std::is_same_v<column_type, cudf::struct_view>, std::unique_ptr<column>>
  operator()(column_view const& column, host_span<column_name_info const> children_names) const
  {
    auto const child_it = cudf::detail::make_counting_transform_iterator(
      0, [&stream = stream_, structs_view = structs_column_view{column}](auto const child_idx) {
        return structs_view.get_sliced_child(child_idx, stream);
      });
    auto col_string = operator()(child_it,
                                 child_it + column.num_children(),
                                 children_names,
                                 column.size(),
                                 struct_row_end_wrap.value(stream_));
    col_string->set_null_mask(cudf::detail::copy_bitmask(column, stream_, mr_),
                              column.null_count());
    return col_string;
  }

  // Table:
  template <typename column_iterator>
  std::unique_ptr<column> operator()(column_iterator column_begin,
                                     column_iterator column_end,
                                     host_span<column_name_info const> children_names,
                                     size_type num_rows,
                                     cudf::string_view const row_end_wrap_value) const
  {
    auto const num_columns = std::distance(column_begin, column_end);
    auto column_names      = make_column_names_column(children_names, num_columns, stream_);
    auto column_names_view = column_names->view();
    std::vector<std::unique_ptr<cudf::column>> str_column_vec;

    // populate vector of string-converted columns:
    //
    auto i_col_begin =
      thrust::make_zip_iterator(thrust::counting_iterator<size_t>(0), column_begin);
    std::transform(
      i_col_begin,
      i_col_begin + num_columns,
      std::back_inserter(str_column_vec),
      [this, &children_names](auto const& i_current_col) {
        auto const i            = thrust::get<0>(i_current_col);
        auto const& current_col = thrust::get<1>(i_current_col);
        // Struct needs children's column names
        if (current_col.type().id() == type_id::STRUCT) {
          return this->template operator()<cudf::struct_view>(current_col,
                                                              children_names.size() > i
                                                                ? children_names[i].children
                                                                : std::vector<column_name_info>{});
        } else if (current_col.type().id() == type_id::LIST) {
          return this->template operator()<cudf::list_view>(current_col,
                                                            children_names.size() > i
                                                              ? children_names[i].children
                                                              : std::vector<column_name_info>{});
        } else {
          return cudf::type_dispatcher<cudf::id_to_type_impl, column_to_strings_fn const&>(
            current_col.type(), *this, current_col);
        }
      });

    // create string table view from str_column_vec:
    //
    auto str_table_ptr  = std::make_unique<cudf::table>(std::move(str_column_vec));
    auto str_table_view = str_table_ptr->view();

    // concatenate columns in each row into one big string column
    // (using null representation and delimiter):
    //
    return struct_to_strings(str_table_view,
                             column_names_view,
                             num_rows,
                             struct_row_begin_wrap.value(stream_),
                             row_end_wrap_value,
                             struct_value_separator.value(stream_),
                             narep,
                             options_.is_enabled_include_nulls(),
                             stream_,
                             cudf::get_current_device_resource_ref());
  }

 private:
  json_writer_options const& options_;
  rmm::cuda_stream_view stream_;
  rmm::device_async_resource_ref mr_;
  string_scalar const narep;  // "null"
  // struct convert constants
  string_scalar const struct_value_separator;  // ","
  string_scalar const struct_row_begin_wrap;   // "{"
  string_scalar const struct_row_end_wrap;     // "}"
  // list converter constants
  string_scalar const list_value_separator;  // ","
  string_scalar const list_row_begin_wrap;   // "["
  string_scalar const list_row_end_wrap;     // "]"
  // bool converter constants
  string_scalar const true_value;
  string_scalar const false_value;
};

}  // namespace

std::unique_ptr<column> make_strings_column_from_host(host_span<std::string const> host_strings,
                                                      rmm::cuda_stream_view stream)
{
  std::string const host_chars =
    std::accumulate(host_strings.begin(), host_strings.end(), std::string(""));
  auto d_chars = cudf::detail::make_device_uvector_async(
    host_chars, stream, cudf::get_current_device_resource_ref());
  std::vector<cudf::size_type> offsets(host_strings.size() + 1, 0);
  std::transform_inclusive_scan(host_strings.begin(),
                                host_strings.end(),
                                offsets.begin() + 1,
                                std::plus<cudf::size_type>{},
                                [](auto& str) { return str.size(); });
  auto d_offsets =
    std::make_unique<cudf::column>(cudf::detail::make_device_uvector_sync(
                                     offsets, stream, cudf::get_current_device_resource_ref()),
                                   rmm::device_buffer{},
                                   0);
  return cudf::make_strings_column(
    host_strings.size(), std::move(d_offsets), d_chars.release(), 0, {});
}

std::unique_ptr<column> make_column_names_column(host_span<column_name_info const> column_names,
                                                 size_type num_columns,
                                                 rmm::cuda_stream_view stream)
{
  std::vector<std::string> unescaped_column_names;
  if (column_names.empty()) {
    std::generate_n(std::back_inserter(unescaped_column_names), num_columns, [v = 0]() mutable {
      return std::to_string(v++);
    });
  } else {
    std::transform(column_names.begin(),
                   column_names.end(),
                   std::back_inserter(unescaped_column_names),
                   [](column_name_info const& name_info) { return name_info.name; });
  }
  auto unescaped_string_col = make_strings_column_from_host(unescaped_column_names, stream);
  auto d_column             = column_device_view::create(*unescaped_string_col, stream);
  return escape_strings_fn{*d_column, true}.get_escaped_strings(
    *unescaped_string_col, stream, cudf::get_current_device_resource_ref());
}

void write_chunked(data_sink* out_sink,
                   strings_column_view const& str_column_view,
                   int const skip_last_chars,
                   json_writer_options const& options,
                   rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(str_column_view.size() > 0, "Unexpected empty strings column.");

  auto const total_num_bytes = str_column_view.chars_size(stream) - skip_last_chars;
  char const* ptr_all_bytes  = str_column_view.chars_begin(stream);

  if (out_sink->is_device_write_preferred(total_num_bytes)) {
    // Direct write from device memory
    out_sink->device_write(ptr_all_bytes, total_num_bytes, stream);
  } else {
    // copy the bytes to host to write them out
    auto const h_bytes = cudf::detail::make_host_vector_sync(
      device_span<char const>(ptr_all_bytes, total_num_bytes), stream);

    out_sink->host_write(h_bytes.data(), total_num_bytes);
  }
}

void write_json_uncompressed(data_sink* out_sink,
                             table_view const& table,
                             json_writer_options const& options,
                             rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  std::vector<column_name_info> user_column_names = [&]() {
    auto const& metadata = options.get_metadata();
    if (metadata.has_value() and not metadata->schema_info.empty()) {
      return metadata->schema_info;
    } else {
      std::vector<column_name_info> names;
      // generate strings 0 to table.num_columns()
      std::transform(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(table.num_columns()),
                     std::back_inserter(names),
                     [](auto i) { return column_name_info{std::to_string(i)}; });
      return names;
    }
  }();
  auto const line_terminator = std::string(options.is_enabled_lines() ? "\n" : ",");
  string_scalar const d_line_terminator_with_row_end{"}" + line_terminator, true, stream};
  string_scalar const d_line_terminator{line_terminator, true, stream};

  // write header: required for non-record oriented output
  // header varies depending on orient.
  // write_chunked_begin(out_sink, table, user_column_names, options, stream, mr);
  // TODO This should go into the write_chunked_begin function
  std::string const list_braces{"[]"};
  string_scalar const d_list_braces{list_braces, true, stream};
  if (!options.is_enabled_lines()) {
    if (out_sink->is_device_write_preferred(1)) {
      out_sink->device_write(d_list_braces.data(), 1, stream);
    } else {
      out_sink->host_write(list_braces.data(), 1);
    }
  }

  if (table.num_rows() > 0) {
    auto n_rows_per_chunk = options.get_rows_per_chunk();

    // This outputs the JSON in row chunks to save memory.
    // Maybe we can use the total_rows*count calculation and a memory threshold
    // instead of an arbitrary chunk count.
    // The entire JSON chunk must fit in CPU memory before writing it out.
    //
    if (n_rows_per_chunk % 8)  // must be divisible by 8
      n_rows_per_chunk += 8 - (n_rows_per_chunk % 8);

    CUDF_EXPECTS(n_rows_per_chunk >= 8, "write_json: invalid chunk_rows; must be at least 8");

    auto num_rows = table.num_rows();
    std::vector<table_view> vector_views;

    if (num_rows <= n_rows_per_chunk) {
      vector_views.push_back(table);
    } else {
      auto const n_chunks = num_rows / n_rows_per_chunk;
      std::vector<size_type> splits(n_chunks);
      thrust::tabulate(splits.begin(), splits.end(), [n_rows_per_chunk](auto idx) {
        return (idx + 1) * n_rows_per_chunk;
      });

      // split table_view into chunks:
      vector_views = cudf::detail::split(table, splits, stream);
    }

    // convert each chunk to JSON:
    column_to_strings_fn converter{options, stream, cudf::get_current_device_resource_ref()};

    for (auto&& sub_view : vector_views) {
      // Skip if the table has no rows
      if (sub_view.num_rows() == 0) continue;
      std::vector<std::unique_ptr<column>> str_column_vec;

      // struct converter for the table
      auto str_concat_col = converter(sub_view.begin(),
                                      sub_view.end(),
                                      user_column_names,
                                      sub_view.num_rows(),
                                      d_line_terminator_with_row_end.value(stream));

      // Needs line_terminator at the end, to separate from next chunk
      bool const include_line_terminator =
        (&sub_view != &vector_views.back()) or options.is_enabled_lines();
      auto const skip_last_chars = (include_line_terminator ? 0 : line_terminator.size());
      write_chunked(out_sink, str_concat_col->view(), skip_last_chars, options, stream);
    }
  } else {
    if (options.is_enabled_lines()) {
      if (out_sink->is_device_write_preferred(1)) {
        out_sink->device_write(d_line_terminator.data(), d_line_terminator.size(), stream);
      } else {
        out_sink->host_write(line_terminator.data(), line_terminator.size());
      }
    }
  }
  // TODO write_chunked_end(out_sink, options, stream, mr);
  if (!options.is_enabled_lines()) {
    if (out_sink->is_device_write_preferred(1)) {
      out_sink->device_write(d_list_braces.data() + 1, 1, stream);
    } else {
      out_sink->host_write(list_braces.data() + 1, 1);
    }
  }
}

void write_json(data_sink* out_sink,
                table_view const& table,
                json_writer_options const& options,
                rmm::cuda_stream_view stream)
{
  if (options.get_compression() != compression_type::NONE) {
    std::vector<char> hbuf;
    auto hbuf_sink_ptr = data_sink::create(&hbuf);
    write_json_uncompressed(hbuf_sink_ptr.get(), table, options, stream);
    stream.synchronize();
    auto comp_hbuf = cudf::io::detail::compress(
      options.get_compression(),
      host_span<uint8_t>(reinterpret_cast<uint8_t*>(hbuf.data()), hbuf.size()),
      stream);
    out_sink->host_write(comp_hbuf.data(), comp_hbuf.size());
    return;
  }
  write_json_uncompressed(out_sink, table, options, stream);
}

}  // namespace cudf::io::json::detail
