/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
 * @file writer_impl.cu
 * @brief cuDF-IO CSV writer class implementation
 */

#include "durations.hpp"

#include "csv_common.hpp"
#include "csv_gpu.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/detail/csv.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cudf {
namespace io {
namespace detail {
namespace csv {

using namespace cudf::io::csv;
using namespace cudf::io;

namespace {

/**
 * @brief Functor to modify a string column for CSV format.
 *
 * If a row contains specific characters, the entire row must be
 * output in double-quotes. Also, if a double-quote appears it
 * must be escaped using a 2nd double-quote.
 */
struct escape_strings_fn {
  column_device_view const d_column;
  string_view const d_delimiter;  // check for column delimiter
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

    constexpr char_utf8 const quote    = '\"';  // check for quote
    constexpr char_utf8 const new_line = '\n';  // and for new-line

    auto const d_str = d_column.element<string_view>(idx);

    // if quote, new-line or a column delimiter appear in the string
    // the entire string must be double-quoted.
    bool const quote_row = thrust::any_of(
      thrust::seq, d_str.begin(), d_str.end(), [d_delimiter = d_delimiter](auto chr) {
        return chr == quote || chr == new_line || chr == d_delimiter[0];
      });

    char* d_buffer    = d_chars ? d_chars + d_offsets[idx] : nullptr;
    offset_type bytes = 0;

    if (quote_row) write_char(quote, d_buffer, bytes);
    for (auto chr : d_str) {
      if (chr == quote) write_char(quote, d_buffer, bytes);
      write_char(chr, d_buffer, bytes);
    }
    if (quote_row) write_char(quote, d_buffer, bytes);

    if (!d_chars) d_offsets[idx] = bytes;
  }
};

struct column_to_strings_fn {
  // compile-time predicate that defines unsupported column types;
  // based on the conditions used for instantiations of individual
  // converters in strings/convert/convert_*.hpp;
  //(this should have been a `variable template`,
  // instead of a static function, but nvcc (10.0)
  // fails to compile var-templs);
  //
  template <typename column_type>
  constexpr static bool is_not_handled()
  {
    // Note: the case (not std::is_same_v<column_type, bool>)
    // is already covered by is_integral)
    //
    return not((std::is_same_v<column_type, cudf::string_view>) ||
               (std::is_integral_v<column_type>) || (std::is_floating_point_v<column_type>) ||
               (cudf::is_fixed_point<column_type>()) || (cudf::is_timestamp<column_type>()) ||
               (cudf::is_duration<column_type>()));
  }

  explicit column_to_strings_fn(csv_writer_options const& options,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
    : options_(options), stream_(stream), mr_(mr)
  {
  }

  // Note: `null` replacement with `na_rep` deferred to `concatenate()`
  // instead of column-wise; might be faster
  //
  // Note: Cannot pass `stream` to detail::<fname> version of <fname> calls below, because they are
  // not exposed in header (see, for example, detail::concatenate(tbl_view, separator, na_rep,
  // stream, mr) is declared and defined in combine.cu); Possible solution: declare `extern`, or
  // just declare a prototype inside `namespace cudf::strings::detail`;

  // bools:
  //
  template <typename column_type>
  std::enable_if_t<std::is_same_v<column_type, bool>, std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    return cudf::strings::detail::from_booleans(
      column, options_.get_true_value(), options_.get_false_value(), stream_, mr_);
  }

  // strings:
  //
  template <typename column_type>
  std::enable_if_t<std::is_same_v<column_type, cudf::string_view>, std::unique_ptr<column>>
  operator()(column_view const& column_v) const
  {
    // handle special characters: {delimiter, '\n', "} in row:
    string_scalar delimiter{std::string{options_.get_inter_column_delimiter()}, true, stream_};

    auto d_column = column_device_view::create(column_v, stream_);
    escape_strings_fn fn{*d_column, delimiter.value(stream_)};
    auto children = cudf::strings::detail::make_strings_children(fn, column_v.size(), stream_, mr_);

    return make_strings_column(column_v.size(),
                               std::move(children.first),
                               std::move(children.second),
                               column_v.null_count(),
                               cudf::detail::copy_bitmask(column_v, stream_, mr_));
  }

  // ints:
  //
  template <typename column_type>
  std::enable_if_t<std::is_integral_v<column_type> && !std::is_same_v<column_type, bool>,
                   std::unique_ptr<column>>
  operator()(column_view const& column) const
  {
    return cudf::strings::detail::from_integers(column, stream_, mr_);
  }

  // floats:
  //
  template <typename column_type>
  std::enable_if_t<std::is_floating_point_v<column_type>, std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    return cudf::strings::detail::from_floats(column, stream_, mr_);
  }

  // fixed point:
  //
  template <typename column_type>
  std::enable_if_t<cudf::is_fixed_point<column_type>(), std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    return cudf::strings::detail::from_fixed_point(column, stream_, mr_);
  }

  // timestamps:
  //
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

    // handle the cases where delimiter / line-terminator can be
    // "-" or ":", in which case we need to add quotes to the format
    //
    std::string delimiter{options_.get_inter_column_delimiter()};
    std::string newline{options_.get_line_terminator()};

    constexpr char const* dash{"-"};
    constexpr char const* colon{":"};
    if (delimiter == dash || newline == dash || delimiter == colon || newline == colon) {
      format = "\"" + format + "\"";
    }

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
    return cudf::io::detail::csv::pandas_format_durations(column, stream_, mr_);
  }

  // unsupported type of column:
  //
  template <typename column_type>
  std::enable_if_t<is_not_handled<column_type>(), std::unique_ptr<column>> operator()(
    column_view const&) const
  {
    CUDF_FAIL("Unsupported column type.");
  }

 private:
  csv_writer_options const& options_;
  rmm::cuda_stream_view stream_;
  rmm::mr::device_memory_resource* mr_;
};
}  // unnamed namespace

// write the header: column names:
//
void write_chunked_begin(data_sink* out_sink,
                         table_view const& table,
                         host_span<std::string const> user_column_names,
                         csv_writer_options const& options,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr)
{
  if (options.is_enabled_include_header()) {
    // need to generate column names if names are not provided
    std::vector<std::string> generated_col_names;
    if (user_column_names.empty()) {
      generated_col_names.resize(table.num_columns());
      thrust::tabulate(generated_col_names.begin(), generated_col_names.end(), [](auto idx) {
        return std::to_string(idx);
      });
    }
    auto const& column_names = user_column_names.empty() ? generated_col_names : user_column_names;
    CUDF_EXPECTS(column_names.size() == static_cast<size_t>(table.num_columns()),
                 "Mismatch between number of column headers and table columns.");

    auto const delimiter  = options.get_inter_column_delimiter();
    auto const terminator = options.get_line_terminator();

    // process header names:
    // - if the header name includes the delimiter or terminator character,
    //   it must be double-quoted
    // - if the header name includes a double-quote, it must be escaped
    //   with a 2nd double-quote
    std::stringstream ss;
    std::transform(column_names.begin(),
                   column_names.end(),
                   std::ostream_iterator<std::string>(ss, std::string{delimiter}.c_str()),
                   [delimiter, terminator](std::string name) {
                     char const quote = '\"';
                     if (name.empty() ||           // no header name
                         name.front() == quote) {  // header already quoted
                       return name;
                     }

                     // escape any quotes
                     size_t pos = 0;
                     while ((pos = name.find(quote, pos)) != name.npos) {
                       name.insert(pos, 1, quote);
                       pos += 2;
                     }

                     // check if overall quotes are required
                     if (std::any_of(name.begin(), name.end(), [&](auto const chr) {
                           return chr == quote || chr == delimiter || chr == terminator.front();
                         })) {
                       name.insert(name.begin(), quote);
                       name.insert(name.end(), quote);
                     }
                     return name;
                   });

    // add line terminator
    std::string header = ss.str();
    if (!header.empty()) {
      header.erase(header.end() - 1);  // remove extra final delimiter
    }
    header.append(terminator);

    out_sink->host_write(header.data(), header.size());
  }
}

void write_chunked(data_sink* out_sink,
                   strings_column_view const& str_column_view,
                   csv_writer_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
{
  // algorithm outline:
  //
  //  for_each(strings_column.begin(), strings_column.end(),
  //           [sink = out_sink](auto str_row) mutable {
  //               auto host_buffer = str_row.host_buffer();
  //               sink->host_write(host_buffer_.data(), host_buffer_.size());
  //           });//or...sink->device_write(device_buffer,...);
  //
  // added line_terminator functionality
  //

  CUDF_EXPECTS(str_column_view.size() > 0, "Unexpected empty strings column.");

  cudf::string_scalar newline{options.get_line_terminator()};
  auto p_str_col_w_nl = cudf::strings::detail::join_strings(str_column_view,
                                                            newline,
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
    thrust::host_vector<char> h_bytes(total_num_bytes);
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_bytes.data(),
                                  ptr_all_bytes,
                                  total_num_bytes * sizeof(char),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();

    out_sink->host_write(h_bytes.data(), total_num_bytes);
  }

  // Needs newline at the end, to separate from next chunk
  if (out_sink->is_device_write_preferred(newline.size())) {
    out_sink->device_write(newline.data(), newline.size(), stream);
  } else {
    out_sink->host_write(options.get_line_terminator().data(),
                         options.get_line_terminator().size());
  }
}

void write_csv(data_sink* out_sink,
               table_view const& table,
               host_span<std::string const> user_column_names,
               csv_writer_options const& options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
{
  // write header: column names separated by delimiter:
  // (even for tables with no rows)
  //
  write_chunked_begin(out_sink, table, user_column_names, options, stream, mr);

  if (table.num_rows() > 0) {
    // no need to check same-size columns constraint; auto-enforced by table_view
    auto n_rows_per_chunk = options.get_rows_per_chunk();
    //
    // This outputs the CSV in row chunks to save memory.
    // Maybe we can use the total_rows*count calculation and a memory threshold
    // instead of an arbitrary chunk count.
    // The entire CSV chunk must fit in CPU memory before writing it out.
    //
    if (n_rows_per_chunk % 8)  // must be divisible by 8
      n_rows_per_chunk += 8 - (n_rows_per_chunk % 8);

    CUDF_EXPECTS(n_rows_per_chunk >= 8, "write_csv: invalid chunk_rows; must be at least 8");

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

    // convert each chunk to CSV:
    //
    column_to_strings_fn converter{options, stream, rmm::mr::get_current_device_resource()};
    for (auto&& sub_view : vector_views) {
      // Skip if the table has no rows
      if (sub_view.num_rows() == 0) continue;
      std::vector<std::unique_ptr<column>> str_column_vec;

      // populate vector of string-converted columns:
      //
      std::transform(sub_view.begin(),
                     sub_view.end(),
                     std::back_inserter(str_column_vec),
                     [converter](auto const& current_col) {
                       return cudf::type_dispatcher(current_col.type(), converter, current_col);
                     });

      // create string table view from str_column_vec:
      //
      auto str_table_ptr  = std::make_unique<cudf::table>(std::move(str_column_vec));
      auto str_table_view = str_table_ptr->view();

      // concatenate columns in each row into one big string column
      // (using null representation and delimiter):
      //
      std::string delimiter_str{options.get_inter_column_delimiter()};
      auto str_concat_col = [&] {
        if (str_table_view.num_columns() > 1)
          return cudf::strings::detail::concatenate(str_table_view,
                                                    delimiter_str,
                                                    options.get_na_rep(),
                                                    strings::separator_on_nulls::YES,
                                                    stream,
                                                    rmm::mr::get_current_device_resource());
        cudf::string_scalar narep{options.get_na_rep()};
        return cudf::strings::detail::replace_nulls(
          str_table_view.column(0), narep, stream, rmm::mr::get_current_device_resource());
      }();

      write_chunked(out_sink, str_concat_col->view(), options, stream, mr);
    }
  }
}

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace cudf
