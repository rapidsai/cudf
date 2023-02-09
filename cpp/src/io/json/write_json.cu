/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <io/csv/durations.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/detail/data_casting.cuh>
#include <cudf/io/detail/json.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/struct_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
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
  offset_type* d_offsets{};
  char* d_chars{};

  __device__ void write_char(char_utf8 chr, char*& d_buffer, offset_type& bytes)
  {
    if (d_buffer)
      d_buffer += cudf::strings::detail::from_char_utf8(chr, d_buffer);
    else
      bytes += cudf::strings::detail::bytes_in_char_utf8(chr);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }

    auto const d_str = d_column.element<string_view>(idx);

    // entire string must be double-quoted.
    constexpr char_utf8 const quote = '\"';  // wrap quotes
    bool constexpr quote_row        = true;

    char* d_buffer    = d_chars ? d_chars + d_offsets[idx] : nullptr;
    offset_type bytes = 0;

    if (quote_row) write_char(quote, d_buffer, bytes);
    for (auto chr : d_str) {
      auto escaped_chars = cudf::io::json::experimental::detail::get_escaped_char(chr);
      if (escaped_chars.first == '\0') {
        write_char(escaped_chars.second, d_buffer, bytes);
      } else {
        write_char(escaped_chars.first, d_buffer, bytes);
        write_char(escaped_chars.second, d_buffer, bytes);
      }
    }
    if (quote_row) write_char(quote, d_buffer, bytes);

    if (!d_chars) d_offsets[idx] = bytes;
  }

  std::unique_ptr<column> get_escaped_strings(column_view const& column_v,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
  {
    auto children =
      cudf::strings::detail::make_strings_children(*this, column_v.size(), stream, mr);

    return make_strings_column(column_v.size(),
                               std::move(children.first),
                               std::move(children.second),
                               column_v.null_count(),
                               cudf::detail::copy_bitmask(column_v, stream, mr));
  }
};

// Struct concatenation.
struct concat_structs_base {
  table_device_view const d_table;
  column_device_view const d_column_names;
  string_view const row_prefix;             //{
  string_view const row_suffix;             //} or }\n for json-lines
  string_view const d_col_separator;        //:
  string_view const d_val_separator;        //,
  string_scalar_device_view const d_narep;  // null
  bool include_nulls = false;
  offset_type* d_offsets{};
  char* d_chars{};

  /**
   * @brief Concatenate each table row to a single output string.
   *
   * This will concatenate the strings from each row of the given table
   * and apply the separator. The null-replacement string `d_narep` is
   * used in place of any string in a row that contains a null entry.
   *
   * @param idx The current row to process
   * @param d_separator String to place in between each column's row
   */
  __device__ void operator()(size_type idx)
  {
    if (!d_narep.is_valid() &&
        thrust::any_of(thrust::seq, d_table.begin(), d_table.end(), [idx](auto const& col) {
          return col.is_null(idx);
        })) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }

    char* d_buffer       = d_chars ? d_chars + d_offsets[idx] : nullptr;
    offset_type bytes    = 0;
    bool write_separator = false;

    if (d_buffer) d_buffer = strings::detail::copy_string(d_buffer, row_prefix);
    bytes += row_prefix.size_bytes();

    for (auto itr = d_table.begin(); itr < d_table.end(); ++itr) {
      auto const col_idx         = thrust::distance(d_table.begin(), itr);
      auto const d_column        = *itr;
      bool const is_null_element = d_column.is_null(idx);
      bool const include_element = (include_nulls == true || !is_null_element);

      if (write_separator && include_element) {
        if (d_buffer) d_buffer = strings::detail::copy_string(d_buffer, d_val_separator);
        bytes += d_val_separator.size_bytes();
        write_separator = false;
      }

      // column_name:
      if (include_element && !d_column_names.is_null(col_idx)) {
        auto const d_name = d_column_names.element<string_view>(col_idx);
        if (d_buffer) d_buffer = strings::detail::copy_string(d_buffer, d_name);
        bytes += d_name.size_bytes();
        if (d_buffer) d_buffer = strings::detail::copy_string(d_buffer, d_col_separator);
        bytes += d_col_separator.size_bytes();
      }

      // write out column's row data (or narep if the row is null)
      if (include_element) {
        auto const d_str = is_null_element ? d_narep.value() : d_column.element<string_view>(idx);
        if (d_buffer) d_buffer = strings::detail::copy_string(d_buffer, d_str);
        bytes += d_str.size_bytes();
      }

      write_separator = write_separator || include_element;
    }
    if (d_buffer) {
      d_buffer = strings::detail::copy_string(d_buffer, row_suffix);
    } else {
      d_offsets[idx] = bytes + row_suffix.size_bytes();
    }
  }
};

/**
 * @brief Concatenate the strings from each row of the given table as structs in JSON string
 *
 * Each row will be struct with field name as column names and values from each column in the table.
 *
 * @param strings_columns Table of strings columns
 * @param column_names Column of names for each column in the table
 * @param row_prefix Prepend this string to each row
 * @param row_suffix  Append this string to each row
 * @param column_name_separator  Separator between column name and value
 * @param value_separator Separator between values
 * @param narep Null-String replacement
 * @param include_nulls Include null string entries in the output
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource to use for device memory allocation.
 * @return New strings column of JSON structs in each row
 */
std::unique_ptr<column> struct_to_strings(table_view const& strings_columns,
                                          column_view const& column_names,
                                          string_view const row_prefix,
                                          string_view const row_suffix,
                                          string_view const column_name_separator,
                                          string_view const value_separator,
                                          string_scalar const& narep,
                                          bool include_nulls,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(column_names.type().id() == type_id::STRING, "Column names must be of type string");
  auto const num_columns = strings_columns.num_columns();
  CUDF_EXPECTS(num_columns == column_names.size(),
               "Number of column names should be equal to number of columns in the table");
  auto const strings_count = strings_columns.num_rows();
  if (strings_count == 0)  // empty begets empty
    return make_empty_column(type_id::STRING);
  // check all columns are of type string
  CUDF_EXPECTS(std::all_of(strings_columns.begin(),
                           strings_columns.end(),
                           [](auto const& c) { return c.type().id() == type_id::STRING; }),
               "All columns must be of type string");

  // Create device views from the strings columns.
  auto d_table        = table_device_view::create(strings_columns, stream);
  auto d_column_names = column_device_view::create(column_names, stream);
  auto d_narep        = get_scalar_device_view(const_cast<string_scalar&>(narep));
  concat_structs_base fn{*d_table,
                         *d_column_names,
                         row_prefix,
                         row_suffix,
                         column_name_separator,
                         value_separator,
                         d_narep,
                         include_nulls};
  auto children = strings::detail::make_strings_children(fn, strings_count, stream, mr);

  // create resulting null mask
  auto [null_mask, null_count] = cudf::detail::valid_if(
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings_count),
    [d_table = *d_table, d_narep] __device__(size_type idx) {
      if (d_narep.is_valid()) return true;
      return !thrust::any_of(
        thrust::seq, d_table.begin(), d_table.end(), [idx](auto col) { return col.is_null(idx); });
    },
    stream,
    mr);

  return make_strings_column(strings_count,
                             std::move(children.first),
                             std::move(children.second),
                             null_count,
                             std::move(null_mask));
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
                                rmm::mr::device_memory_resource* mr)
    : options_(options), stream_(stream), mr_(mr), narep(options.get_na_rep())
  {
  }

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
    return cudf::strings::detail::from_booleans(
      column, options_.get_true_value(), options_.get_false_value(), stream_, mr_);
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
      strings_column_view(column_view{data_type{type_id::STRING}, 0, nullptr}),
      stream_,
      mr_);
  }

  template <typename column_type>
  std::enable_if_t<cudf::is_duration<column_type>(), std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    auto duration_string = cudf::io::detail::csv::pandas_format_durations(column, stream_, mr_);
    auto quotes = make_column_from_scalar(string_scalar{"\""}, column.size(), stream_, mr_);
    return cudf::strings::detail::concatenate(
      table_view{{quotes->view(), duration_string->view(), quotes->view()}},
      string_scalar(""),
      string_scalar("", false),
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
    auto list_string           = [&]() {
      auto child_string = [&]() {
        if (child_view.type().id() == type_id::STRUCT) {
          return (*this).template operator()<cudf::struct_view>(
            child_view,
            children_names.size() > child_index ? children_names[child_index].children
                                                          : std::vector<column_name_info>{});
        } else if (child_view.type().id() == type_id::LIST) {
          return (*this).template operator()<cudf::list_view>(
            child_view,
            children_names.size() > child_index ? children_names[child_index].children
                                                          : std::vector<column_name_info>{});
        } else {
          return cudf::type_dispatcher(child_view.type(), *this, child_view);
        }
      }();
      auto const list_child_string =
        column_view(column.type(),
                    column.size(),
                    column.head(),
                    column.null_mask(),
                    column.null_count(),
                    column.offset(),
                    {lists_column_view(column).offsets(), child_string->view()});
      return strings::detail::join_list_elements(lists_column_view(list_child_string),
                                                 string_scalar{","},
                                                 narep,
                                                 strings::separator_on_nulls::YES,
                                                 strings::output_if_empty_list::EMPTY_STRING,
                                                 stream_,
                                                 mr_);
    }();
    // create column with "[", "]" to wrap around list string
    auto prepend = make_column_from_scalar(string_scalar{"["}, column.size(), stream_, mr_);
    auto append  = make_column_from_scalar(string_scalar{"]"}, column.size(), stream_, mr_);
    return cudf::strings::detail::concatenate(
      table_view{{prepend->view(), list_string->view(), append->view()}},
      string_scalar(""),
      string_scalar("", false),
      strings::separator_on_nulls::YES,
      stream_,
      mr_);
  }

  // structs:
  template <typename column_type>
  std::enable_if_t<std::is_same_v<column_type, cudf::struct_view>, std::unique_ptr<column>>
  operator()(column_view const& column, host_span<column_name_info const> children_names) const
  {
    auto col_string = operator()(column.child_begin(), column.child_end(), children_names);
    col_string->set_null_mask(cudf::detail::copy_bitmask(column, stream_, mr_),
                              column.null_count());
    return col_string;
  }

  // Table:
  template <typename column_iterator>
  std::unique_ptr<column> operator()(column_iterator column_begin,
                                     column_iterator column_end,
                                     host_span<column_name_info const> children_names) const
  {
    auto const num_columns = std::distance(column_begin, column_end);
    auto column_names      = make_column_names_column(children_names, num_columns, stream_);
    auto column_names_view = column_names->view();
    std::vector<std::unique_ptr<cudf::column>> str_column_vec;

    // populate vector of string-converted columns:
    //
    auto i_col_begin =
      thrust::make_zip_iterator(thrust::counting_iterator<size_t>(0), column_begin);
    std::transform(i_col_begin,
                   i_col_begin + num_columns,
                   std::back_inserter(str_column_vec),
                   [this, &children_names](auto const& i_current_col) {
                     auto const i            = thrust::get<0>(i_current_col);
                     auto const& current_col = thrust::get<1>(i_current_col);
                     // Struct needs children's column names
                     if (current_col.type().id() == type_id::STRUCT) {
                       return (*this).template operator()<cudf::struct_view>(
                         current_col,
                         children_names.size() > i ? children_names[i].children
                                                   : std::vector<column_name_info>{});
                     } else if (current_col.type().id() == type_id::LIST) {
                       return (*this).template operator()<cudf::list_view>(
                         current_col,
                         children_names.size() > i ? children_names[i].children
                                                   : std::vector<column_name_info>{});
                     } else {
                       return cudf::type_dispatcher(current_col.type(), *this, current_col);
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
                             row_begin_wrap.value(stream_),
                             row_end_wrap.value(stream_),
                             column_seperator.value(stream_),
                             value_seperator.value(stream_),
                             narep,
                             options_.is_enabled_include_nulls(),
                             stream_,
                             rmm::mr::get_current_device_resource());
  }

 private:
  json_writer_options const& options_;
  rmm::cuda_stream_view stream_;
  rmm::mr::device_memory_resource* mr_;
  string_scalar const column_seperator{":"};
  string_scalar const value_seperator{","};
  string_scalar const row_begin_wrap{"{"};
  string_scalar const row_end_wrap{"}"};
  string_scalar const narep;
};

}  // namespace

std::unique_ptr<column> make_strings_column_from_host(host_span<std::string const> host_strings,
                                                      rmm::cuda_stream_view stream)
{
  std::string const host_chars =
    std::accumulate(host_strings.begin(), host_strings.end(), std::string(""));
  auto d_chars = cudf::detail::make_device_uvector_async(host_chars, stream);
  std::vector<cudf::size_type> offsets(host_strings.size() + 1, 0);
  std::transform_inclusive_scan(host_strings.begin(),
                                host_strings.end(),
                                offsets.begin() + 1,
                                std::plus<cudf::size_type>{},
                                [](auto& str) { return str.size(); });
  auto d_offsets = cudf::detail::make_device_uvector_sync(offsets, stream);
  return cudf::make_strings_column(
    host_strings.size(), std::move(d_offsets), std::move(d_chars), {}, 0);
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
  return escape_strings_fn{*d_column}.get_escaped_strings(
    *unescaped_string_col, stream, rmm::mr::get_current_device_resource());
}

void write_chunked(data_sink* out_sink,
                   strings_column_view const& str_column_view,
                   std::string const& line_terminator,
                   json_writer_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(str_column_view.size() > 0, "Unexpected empty strings column.");

  string_scalar d_line_terminator{line_terminator};
  auto p_str_col_w_nl = cudf::strings::detail::join_strings(str_column_view,
                                                            d_line_terminator,
                                                            string_scalar("", false),
                                                            stream,
                                                            rmm::mr::get_current_device_resource());
  strings_column_view strings_column{p_str_col_w_nl->view()};

  auto total_num_bytes      = strings_column.chars_size();
  char const* ptr_all_bytes = strings_column.chars_begin();

  if (out_sink->is_device_write_preferred(total_num_bytes)) {
    // Direct write from device memory
    out_sink->device_write(ptr_all_bytes, total_num_bytes, stream);
  } else {
    // copy the bytes to host to write them out
    auto const h_bytes = cudf::detail::make_host_vector_sync(
      device_span<char const>(ptr_all_bytes, total_num_bytes), stream);

    out_sink->host_write(h_bytes.data(), total_num_bytes);
  }

  // Needs newline at the end, to separate from next chunk
  if (options.is_enabled_lines()) {
    if (out_sink->is_device_write_preferred(d_line_terminator.size())) {
      out_sink->device_write(d_line_terminator.data(), d_line_terminator.size(), stream);
    } else {
      out_sink->host_write(line_terminator.data(), line_terminator.size());
    }
  }
}

void write_json(data_sink* out_sink,
                table_view const& table,
                json_writer_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr)
{
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
  string_scalar d_line_terminator{line_terminator};

  // write header: required for non-record oriented output
  // header varies depending on orient.
  // write_chunked_begin(out_sink, table, user_column_names, options, stream, mr);
  // TODO This should go into the write_chunked_begin function
  std::string const list_braces{"[]"};
  string_scalar d_list_braces{list_braces};
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
    column_to_strings_fn converter{options, stream, rmm::mr::get_current_device_resource()};

    for (auto&& sub_view : vector_views) {
      // Skip if the table has no rows
      if (sub_view.num_rows() == 0) continue;
      std::vector<std::unique_ptr<column>> str_column_vec;

      // struct converter for the table
      auto str_concat_col = converter(sub_view.begin(), sub_view.end(), user_column_names);

      write_chunked(out_sink, str_concat_col->view(), line_terminator, options, stream, mr);
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

}  // namespace cudf::io::json::detail
