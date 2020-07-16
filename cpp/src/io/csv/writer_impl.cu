/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "writer_impl.hpp"

#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>

#include <cudf/utilities/traits.hpp>

#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>

#include <cudf/strings/combine.hpp>
#include <cudf/strings/replace.hpp>

#include <strings/utilities.cuh>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <sstream>
#include <type_traits>
#include <utility>

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/modify_strings.cuh>

namespace cudf {
namespace io {
namespace detail {
namespace csv {

namespace {  // anonym.
// helpers:

using namespace cudf::strings;

// predicate to determine if a given string_view contains special characters:
//{"\"", "\n", <delimiter>}
//
struct predicate_special_chars {
  explicit predicate_special_chars(string_scalar const& delimiter, cudaStream_t stream = 0)
    : delimiter_(delimiter.value(stream))
  {
  }

  __device__ bool operator()(string_view const& str_view) const
  {
    // if (any_of{"\"", "\n", <delimiter>} )
    //
    constexpr char const* quote_str   = "\"";
    constexpr char const* newline_str = "\n";
    constexpr size_type len1byte{1};

    if ((str_view.find(quote_str, len1byte) >= 0) || (str_view.find(newline_str, len1byte) >= 0) ||
        (str_view.find(delimiter_) >= 0)) {
      return true;
    } else {
      return false;
    }
  }

 private:
  string_view delimiter_;
};

struct probe_special_chars {
  probe_special_chars(column_device_view const d_column, predicate_special_chars const& predicate)
    : d_column_(d_column), predicate_(predicate)
  {
  }

  __device__ int32_t operator()(size_type idx) const
  {
    if (d_column_.is_null(idx)) {
      return 0;  // null string, so no-op
    }

    string_view d_str = d_column_.template element<string_view>(idx);

    if (predicate_(d_str)) {
      constexpr char const quote_char = '\"';

      // count number of quotes "\""
      size_type num_quotes =
        thrust::count_if(thrust::seq, d_str.begin(), d_str.end(), [quote_char](char_utf8 chr) {
          return chr == quote_char;
        });
      return d_str.size_bytes() + num_quotes + 2;
    } else {
      return d_str.size_bytes();
    }
  }

 private:
  column_device_view const d_column_;
  predicate_special_chars predicate_;
};

struct modify_special_chars {
  modify_special_chars(column_device_view const d_column,
                       int32_t const* d_offsets,
                       char* d_chars,
                       predicate_special_chars const& predicate)
    : d_column_(d_column), d_offsets_(d_offsets), d_chars_(d_chars), predicate_(predicate)
  {
  }

  __device__ int32_t operator()(size_type idx)
  {
    using namespace cudf::strings::detail;

    if (d_column_.is_null(idx)) {
      return 0;  // null string, so no-op
    }

    string_view d_str        = d_column_.template element<string_view>(idx);
    size_type str_size_bytes = d_str.size_bytes();

    char* d_buffer = get_output_ptr(idx);
    // assert( d_buffer != nullptr );

    if (predicate_(d_str)) {
      constexpr char const quote_char   = '\"';
      constexpr char const* quote_str   = "\"";
      constexpr char const* str_2quotes = "\"\"";

      size_type len1quote{1};
      size_type len2quotes{2};

      // modify d_str by duplicating all 2bl quotes
      // and surrounding whole string by 2bl quotes:
      //
      // pre-condition: `d_str` is _not_ modified by `d_buffer` manipulation
      // because it's a copy of `idx` entry in `d_column_`
      //(since `d_column` is const)
      //
      d_buffer = copy_and_increment(d_buffer, quote_str, len1quote);  // add the quote prefix

      for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
        char_utf8 the_chr = *itr;

        if (the_chr == quote_char) {
          d_buffer = copy_and_increment(d_buffer, str_2quotes, len2quotes);  // double the quote;
        } else {
          d_buffer += from_char_utf8(the_chr, d_buffer);
        }
      }

      d_buffer = copy_and_increment(d_buffer, quote_str, len1quote);  // add the quote suffix;
    } else {
      // copy the source string unmodified:
      //(pass-through)
      //
      memcpy(get_output_ptr(idx), d_str.data(), str_size_bytes);
    }
    return 0;
  }

  __device__ char* get_output_ptr(size_type idx)
  {
    return d_chars_ && d_offsets_ ? d_chars_ + d_offsets_[idx] : nullptr;
  }

 private:
  column_device_view const d_column_;
  int32_t const* d_offsets_;
  char* d_chars_;
  predicate_special_chars predicate_;
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
  constexpr static bool is_not_handled(void)
  {
    // Note: the case (not std::is_same<column_type, bool>::value)
    // is already covered by is_integral)
    //
    return not((std::is_same<column_type, cudf::string_view>::value) ||
               (std::is_integral<column_type>::value) ||
               (std::is_floating_point<column_type>::value) || (cudf::is_timestamp<column_type>()));
  }

  explicit column_to_strings_fn(writer_options const& options,
                                rmm::mr::device_memory_resource* mr = nullptr,
                                cudaStream_t stream                 = nullptr)
    : options_(options), mr_(mr), stream_(stream)
  {
  }

  // Note: `null` replacement with `na_rep` deferred to `concatenate()`
  // instead of column-wise; might be faster
  //
  // Note: Cannot pass `stream` to detail::<fname> version of <fname> calls below, because they are
  // not exposed in header (see, for example, detail::concatenate(tbl_view, separator, na_rep, mr,
  // stream) is declared and defined in combine.cu); Possible solution: declare `extern`, or just
  // declare a prototype inside `namespace cudf::strings::detail`;

  // bools:
  //
  template <typename column_type>
  std::enable_if_t<std::is_same<column_type, bool>::value, std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    auto conv_col_ptr =
      cudf::strings::from_booleans(column, options_.true_value(), options_.false_value(), mr_);

    return conv_col_ptr;
  }

  // strings:
  //
  template <typename column_type>
  std::enable_if_t<std::is_same<column_type, cudf::string_view>::value, std::unique_ptr<column>>
  operator()(column_view const& column_v) const
  {
    using namespace cudf::strings::detail;

    // handle special characters: {delimiter, '\n', "} in row:
    //
    // algorithm outline:
    //
    // target = "\"";
    // repl = ""\"\";
    //
    // str_column_ref = {};
    // for each str_row: column_v {
    //    if ((not null str_row) &&
    //        (str_row.find("\n") || str_row.find("\"") || str_row.find(delimiter) ))
    //        str_column_modified = modify(str_row);
    // where modify() = duplicate the double quotes, if any; add 2bl quotes prefix/suffix;
    //}
    //
    std::string delimiter{options_.inter_column_delimiter()};
    predicate_special_chars pred{delimiter, stream_};

    return modify_strings<probe_special_chars, modify_special_chars>(column_v, mr_, stream_, pred);
  }

  // ints:
  //
  template <typename column_type>
  std::enable_if_t<std::is_integral<column_type>::value && !std::is_same<column_type, bool>::value,
                   std::unique_ptr<column>>
  operator()(column_view const& column) const
  {
    auto conv_col_ptr = cudf::strings::from_integers(column, mr_);

    return conv_col_ptr;
  }

  // floats:
  //
  template <typename column_type>
  std::enable_if_t<std::is_floating_point<column_type>::value, std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    auto conv_col_ptr = cudf::strings::from_floats(column, mr_);

    return conv_col_ptr;
  }

  // timestamps:
  //
  template <typename column_type>
  std::enable_if_t<cudf::is_timestamp<column_type>(), std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    std::string format{"%Y-%m-%dT%H:%M:%SZ"};  // same as default for `from_timestamp`

    // handle the cases where delimiter / line-terminator can be
    // "-" or ":", in which case they are to be dropped from the format:
    //
    std::string delimiter{options_.inter_column_delimiter()};
    std::string newline{options_.line_terminator()};

    constexpr char const* dash{"-"};
    constexpr char const* colon{":"};
    if (delimiter == dash || newline == dash) {
      format.erase(std::remove(format.begin(), format.end(), dash[0]), format.end());
    }

    if (delimiter == colon || newline == colon) {
      format.erase(std::remove(format.begin(), format.end(), colon[0]), format.end());
    }

    auto conv_col_ptr = cudf::strings::from_timestamps(column, format, mr_);

    return conv_col_ptr;
  }

  // unsupported type of column:
  //
  template <typename column_type>
  std::enable_if_t<is_not_handled<column_type>(), std::unique_ptr<column>> operator()(
    column_view const& column) const
  {
    CUDF_FAIL("Unsupported column type.");
  }

 private:
  writer_options const& options_;
  rmm::mr::device_memory_resource* mr_;
  cudaStream_t stream_;
};
}  // unnamed namespace

// Forward to implementation
writer::writer(std::unique_ptr<data_sink> sink,
               writer_options const& options,
               rmm::mr::device_memory_resource* mr)
  : _impl(std::make_unique<impl>(std::move(sink), options, mr))
{
}

// Destructor within this translation unit
writer::~writer() = default;

writer::impl::impl(std::unique_ptr<data_sink> sink,
                   writer_options const& options,
                   rmm::mr::device_memory_resource* mr)
  : out_sink_(std::move(sink)), mr_(mr), options_(options)
{
}

// write the header: column names:
//
void writer::impl::write_chunked_begin(table_view const& table,
                                       const table_metadata* metadata,
                                       cudaStream_t stream)
{
  if ((metadata != nullptr) && (options_.include_header())) {
    CUDF_EXPECTS(metadata->column_names.size() == static_cast<size_t>(table.num_columns()),
                 "Mismatch between number of column headers and table columns.");

    std::string delimiter_str{options_.inter_column_delimiter()};

    // avoid delimiter after last element:
    //
    std::stringstream ss;
    std::copy(metadata->column_names.begin(),
              metadata->column_names.end() - 1,
              std::ostream_iterator<std::string>(ss, delimiter_str.c_str()));
    ss << metadata->column_names.back() << options_.line_terminator();

    out_sink_->host_write(ss.str().data(), ss.str().size());
  }
}

void writer::impl::write_chunked(strings_column_view const& str_column_view,
                                 const table_metadata* metadata,
                                 cudaStream_t stream)
{
  // algorithm outline:
  //
  //  for_each(strings_column.begin(), strings_column.end(),
  //           [sink = out_sink_](auto str_row) mutable {
  //               auto host_buffer = str_row.host_buffer();
  //               sink->host_write(host_buffer_.data(), host_buffer_.size());
  //           });//or...sink->device_write(device_buffer,...);
  //
  // added line_terminator functionality
  //

  CUDF_EXPECTS(str_column_view.size() > 0, "Unexpected empty strings column.");

  cudf::string_scalar newline{options_.line_terminator()};
  auto p_str_col_w_nl = cudf::strings::join_strings(str_column_view, newline);
  strings_column_view strings_column{std::move(p_str_col_w_nl->view())};

  auto total_num_bytes      = strings_column.chars_size();
  char const* ptr_all_bytes = strings_column.chars().data<char>();

  if (out_sink_->supports_device_write()) {
    // host algorithm call, but the underlying call
    // is a device_write taking a device buffer;
    //
    out_sink_->device_write(ptr_all_bytes, total_num_bytes, stream);
    out_sink_->device_write(newline.data(),
                            newline.size(),
                            stream);  // needs newline at the end, to separate from next chunk
  } else {
    // no device write possible;
    //
    // copy the bytes to host, too:
    //
    thrust::host_vector<char> h_bytes(total_num_bytes);
    CUDA_TRY(cudaMemcpyAsync(h_bytes.data(),
                             ptr_all_bytes,
                             total_num_bytes * sizeof(char),
                             cudaMemcpyDeviceToHost,
                             stream));

    CUDA_TRY(cudaStreamSynchronize(stream));

    // host algorithm call, where the underlying call
    // is also host_write taking a host buffer;
    //
    char const* ptr_h_bytes = h_bytes.data();
    out_sink_->host_write(ptr_h_bytes, total_num_bytes);
    out_sink_->host_write(
      options_.line_terminator().data(),
      options_.line_terminator().size());  // needs newline at the end, to separate from next chunk
  }
}

void writer::impl::write(table_view const& table,
                         const table_metadata* metadata,
                         cudaStream_t stream)
{
  CUDF_EXPECTS(table.num_columns() > 0, "Empty table.");

  // write header: column names separated by delimiter:
  // (even for tables with no rows)
  //
  write_chunked_begin(table, metadata, stream);

  if (table.num_rows() > 0) {
    // no need to check same-size columns constraint; auto-enforced by table_view
    auto n_rows_per_chunk = options_.rows_per_chunk();
    //
    // This outputs the CSV in row chunks to save memory.
    // Maybe we can use the total_rows*count calculation and a memory threshold
    // instead of an arbitrary chunk count.
    // The entire CSV chunk must fit in CPU memory before writing it out.
    //
    if (n_rows_per_chunk % 8)  // must be divisible by 8
      n_rows_per_chunk += 8 - (n_rows_per_chunk % 8);

    CUDF_EXPECTS(n_rows_per_chunk >= 8, "write_csv: invalid chunk_rows; must be at least 8");

    auto exec = rmm::exec_policy(stream);

    auto num_rows = table.num_rows();
    std::vector<table_view> vector_views;

    if (num_rows <= n_rows_per_chunk) {
      vector_views.push_back(table);
    } else {
      std::vector<size_type> splits;
      auto n_chunks = num_rows / n_rows_per_chunk;
      splits.resize(n_chunks);

      rmm::device_vector<size_type> d_splits(n_chunks, n_rows_per_chunk);
      thrust::inclusive_scan(exec->on(stream), d_splits.begin(), d_splits.end(), d_splits.begin());

      CUDA_TRY(cudaMemcpyAsync(splits.data(),
                               d_splits.data().get(),
                               n_chunks * sizeof(size_type),
                               cudaMemcpyDeviceToHost,
                               stream));

      CUDA_TRY(cudaStreamSynchronize(stream));

      // split table_view into chunks:
      //
      vector_views = cudf::split(table, splits);
    }

    // convert each chunk to CSV:
    //
    column_to_strings_fn converter{options_, mr_};
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
      //(using null representation and delimiter):
      //
      std::string delimiter_str{options_.inter_column_delimiter()};
      auto str_concat_col =
        cudf::strings::concatenate(str_table_view, delimiter_str, options_.na_rep(), mr_);

      write_chunked(str_concat_col->view(), metadata, stream);
    }
  }

  // finalize (no-op, for now, but offers a hook for future extensions):
  //
  write_chunked_end(table, metadata, stream);
}

void writer::write_all(table_view const& table, const table_metadata* metadata, cudaStream_t stream)
{
  _impl->write(table, metadata, stream);
}

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace cudf
