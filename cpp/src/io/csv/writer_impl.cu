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
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>

#include <cudf/strings/replace.hpp>
#include <cudf/strings/combine.hpp>

#include <algorithm>
#include <cstring>
#include <utility>
#include <type_traits>
#include <iterator>
#include <sstream>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

namespace cudf {
namespace experimental {
namespace io {
namespace detail {
namespace csv {

namespace {//unnammed:
//helpers:
  
struct column_to_strings_fn
{
  //compile-time predicate that defines unsupported column types;
  //based on the conditions used for instantiations of individual
  //converters in strings/convert/convert_*.hpp;
  //(this should have been a `variable template`,
  // instead of a static function, but nvcc (10.0)
  // fails to compile var-templs);
  //
  template<typename column_type>
  constexpr static bool is_not_handled(void)
  {
    return( //(!std::is_same<column_type, bool>::value) && <- case covered by is_integral
            (!std::is_same<column_type, cudf::string_view>::value) &&
            (!std::is_integral<column_type>::value) &&
            (!std::is_floating_point<column_type>::value) &&
            (!cudf::is_timestamp<column_type>()) );
  }
  
  explicit column_to_strings_fn(writer_options const& options,
                                rmm::mr::device_memory_resource* mr = nullptr):
    options_(options),
    mr_(mr)
  {
  }

  //Note: Done: `null` replacement with `na_rep` defered to `concatenate()`
  //instead of column-wise; might be faster
  //
  //Note: Cannot pass `stream` to detail::<fname> version of <fname> calls below, because they are not exposed in header (see, for example, detail::concatenate(tbl_view, separator, na_rep, mr, stream) is declared and defined in combine.cu);
  //Possible solution: declare `extern`, or just declare a prototype inside `namespace cudf::strings::detail`;

  //bools:
  //
  template<typename column_type>
  std::enable_if_t<std::is_same<column_type, bool>::value,
                   std::unique_ptr<column>>
  operator()(column_view const& column) const
  {
    auto conv_col_ptr = cudf::strings::from_booleans(column,
                                                     options_.true_value(),
                                                     options_.false_value(),
                                                     mr_);

    return conv_col_ptr;

    //null replacement could be done here, but probably more efficient
    //to defer it until concatenate() call:
    //
    // strings_column_view strings_converted{std::move(*conv_col_ptr)};
    // return  cudf::strings::replace_nulls(strings_converted,
    //                                      options_.na_rep() ,
    //                                      mr_);
  }

  //strings:
  //
  template<typename column_type>
  std::enable_if_t<std::is_same<column_type, cudf::string_view>::value,
                   std::unique_ptr<column>>
  operator()(column_view const& column_v) const
  {
    //_not_ just pass through:
    //TODO: must handle special characters: {delimiter, '\n', "} in row:
    //
    // algorithm outline:
    //
    // target = "\"";
    // repl = ""\"\";
    //
    // /* "slice" the part of interest: */
    // str_column_ref = {};
    // for each str_row: column_v {
    //    if ((not null str_row) &&
    //        (str_row.find("\n") || str_row.find("\"") || str_row.find(delimiter) ))        str_column_ref.append(ref(str_row));
    // cudf::strings::replace(str_column_ref, target, repl);
    // prepend(str_column_ref, target); //?
    // append(str_column_ref, target);  //?
    //}
    //
    column col{column_v};
    return std::make_unique<column>(std::move(col));//TODO: look at more efficient way to return a unique_ptr<column> from a column_view...
    
    //null replacement could be done here, but probably more efficient
    //to defer it until concatenate() call:
    //
    // return cudf::strings::replace_nulls(column,
    //                                     options_.na_rep() ,
    //                                     mr_);
  }

  //ints:
  //
  template<typename column_type>
  std::enable_if_t<std::is_integral<column_type>::value && !std::is_same<column_type, bool>::value,
                   std::unique_ptr<column>>
  operator()(column_view const& column) const
  {
    
    auto conv_col_ptr = cudf::strings::from_integers(column,
                                                     mr_);

    return conv_col_ptr;

    //null replacement could be done here, but probably more efficient
    //to defer it until concatenate() call:
    //
    // strings_column_view strings_converted{std::move(*conv_col_ptr)};
    // return cudf::strings::replace_nulls(strings_converted,
    //                                     options_.na_rep() ,
    //                                     mr_);
  }

  //floats:
  //
  template<typename column_type>
  std::enable_if_t<std::is_floating_point<column_type>::value,
                   std::unique_ptr<column>>
  operator()(column_view const& column) const
  {
    auto conv_col_ptr = cudf::strings::from_floats(column,
                                                   mr_);

    return conv_col_ptr;

    //null replacement could be done here, but probably more efficient
    //to defer it until concatenate() call:
    //
    // strings_column_view strings_converted{std::move(*conv_col_ptr)};
    // return cudf::strings::replace_nulls(strings_converted,
    //                                     options_.na_rep() ,
    //                                     mr_);
  }

  //timestamps:
  //
  template<typename column_type>
  std::enable_if_t<cudf::is_timestamp<column_type>(),
                   std::unique_ptr<column>>
  operator()(column_view const& column) const
  {
    std::string format{"%Y-%m-%dT%H:%M:%SZ"};//same as default for `from_timestamp`
    auto conv_col_ptr = cudf::strings::from_timestamps(column,
                                                       format,
                                                       mr_);

    return conv_col_ptr;

    //null replacement could be done here, but probably more efficient
    //to defer it until concatenate() call:
    //
    // strings_column_view strings_converted{std::move(*conv_col_ptr)};
    // return cudf::strings::replace_nulls(strings_converted,
    //                                     options_.na_rep() ,
    //                                     mr_);
  }


  //unsupported type of column:
  //
  template<typename column_type>
  std::enable_if_t<is_not_handled<column_type>(),
                   std::unique_ptr<column>>
  operator()(column_view const& column) const
  {
    CUDF_FAIL("Unsupported column type.");
  }
private:
  writer_options const& options_;
  rmm::mr::device_memory_resource* mr_;
};
} // unnamed namespace

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
                   writer_options const &options,
                   rmm::mr::device_memory_resource *mr):
  out_sink_(std::move(sink)),
  mr_(mr),
  options_(options)
{
}

// write the header: column names:
//
void writer::impl::write_chunked_begin(table_view const& table,
                                       const table_metadata *metadata,
                                       cudaStream_t stream)
{
  if ( options_.include_header() )
    {
      CUDF_EXPECTS( metadata != nullptr, "Unexpected null metadata."); 
      CUDF_EXPECTS( metadata->column_names.size() == static_cast<size_t>(table.num_columns()),
                    "Mismatch between number of column headers and table columns.");

      std::string delimiter_str{options_.inter_column_delimiter()};

      //avoid delimiter after last element:
      //
      std::stringstream ss;
      std::copy(metadata->column_names.begin(), metadata->column_names.end()-1,
                std::ostream_iterator<std::string>(ss, delimiter_str.c_str()));
      ss<<metadata->column_names.back();

      out_sink_->host_write(ss.str().data(), ss.str().size());
    }
}


void writer::impl::write_chunked(strings_column_view const& strings_column,
                                 const table_metadata *metadata,
                                 cudaStream_t stream)
{
  //algorithm outline:
  //
  //  for_each(strings_column.begin(), strings_column.end(),
  //           [sink = out_sink_](auto str_row) mutable {
  //               auto host_buffer = str_row.host_buffer();
  //               sink->host_write(host_buffer_.data(), host_buffer_.size());
  //           });//or...sink->device_write(device_buffer,...);

  auto pair_buff_offsets = cudf::strings::create_offsets(strings_column, stream, mr_);

  auto num_rows = strings_column.size();
  decltype(num_rows) num_offsets = pair_buff_offsets.second.size();

  CUDF_EXPECTS( num_rows == num_offsets, "Unexpected discrepancy between number of offsets and number of rows.");

  
  auto total_num_bytes = pair_buff_offsets.first.size();
  char const* ptr_all_bytes = pair_buff_offsets.first.data().get();

  rmm::device_vector<size_type> d_row_sizes(num_rows);

  //extended device lambdas called inside a member function
  //of a nested classes need:
  //(1) the nested class to be declared public;
  //(2) the calling member function to be public;
  //(otherwise known compiler error)
  //
  //extract row sizes in bytes from the offsets:
  //
  auto exec = rmm::exec_policy(stream);
  thrust::transform(exec->on(stream),
                    thrust::make_counting_iterator<size_type>(0), thrust::make_counting_iterator<size_type>(num_offsets),
                    d_row_sizes.begin(), 
                    [ptr_all_bytes, total_num_bytes, num_offsets] __device__ (auto row_index) {
                      return row_index < num_offsets-1 ? ptr_all_bytes[row_index+1] - ptr_all_bytes[row_index] : total_num_bytes - ptr_all_bytes[row_index]; 
                    });

  //copy offsets to host:
  //
  thrust::host_vector<size_type> h_offsets(num_offsets);
  CUDA_TRY(cudaMemcpyAsync(h_offsets.data(), pair_buff_offsets.second.data().get(), 
                           num_offsets*sizeof(size_type), cudaMemcpyDeviceToHost,
                           stream));

  //copy sizes to host:
  //
  thrust::host_vector<size_type> h_row_sizes(num_rows);
  CUDA_TRY(cudaMemcpyAsync(h_row_sizes.data(), d_row_sizes.data().get(), 
                           num_rows*sizeof(size_type), cudaMemcpyDeviceToHost,
                           stream));

  CUDA_TRY(cudaStreamSynchronize(stream));

  if (out_sink_->supports_device_write()) {
    //host algorithm call, but the underlying call
    //is a device_write taking a device buffer;
    //
    thrust::transform(thrust::host,
                      h_offsets.begin(), h_offsets.end(),
                      h_row_sizes.begin(),
                      thrust::make_discard_iterator(),//discard output
                      [&sink = out_sink_, ptr_all_bytes, stream] (size_type offset_indx, size_type row_sz) mutable {
                        sink->device_write(ptr_all_bytes + offset_indx,
                                           row_sz,
                                           stream);
                        return 0;//discarded (but necessary)
                      });
  } else {
    //no device write possible;
    //
    //copy the bytes to host, too:
    //
    thrust::host_vector<char> h_bytes(total_num_bytes);
    CUDA_TRY(cudaMemcpyAsync(h_bytes.data(), ptr_all_bytes, 
                             total_num_bytes*sizeof(char),
                             cudaMemcpyDeviceToHost,
                             stream));

    CUDA_TRY(cudaStreamSynchronize(stream));

    //host algorithm call, where the underlying call
    //is also host_write taking a host buffer;
    //
    char const* ptr_h_bytes = h_bytes.data();
    thrust::transform(thrust::host,
                      h_offsets.begin(), h_offsets.end(),
                      h_row_sizes.begin(),
                      thrust::make_discard_iterator(),//discard output
                      [&sink = out_sink_, ptr_h_bytes, stream] (size_type offset_indx, size_type row_sz) mutable {
                        sink->host_write(ptr_h_bytes + offset_indx,
                                         row_sz);
                        return 0;//discarded (but necessary)
                      });
  }
}

  
void writer::impl::write(table_view const &table,
                         const table_metadata *metadata,
                         cudaStream_t stream) {
  CUDF_EXPECTS( table.num_columns() > 0 && table.num_rows() > 0, "Empty table." );
  

  //write header: column names separated by delimiter:
  //
  write_chunked_begin(table, metadata, stream);
  
  //no need to check same-size columns constraint; auto-enforced by table_view
  auto n_rows_per_chunk = options_.rows_per_chunk();
  //
  // This outputs the CSV in row chunks to save memory.
  // Maybe we can use the total_rows*count calculation and a memory threshold
  // instead of an arbitrary chunk count.
  // The entire CSV chunk must fit in CPU memory before writing it out.
  //
  if( n_rows_per_chunk % 8 ) // must be divisible by 8
    n_rows_per_chunk += 8 - (n_rows_per_chunk % 8);
  CUDF_EXPECTS( n_rows_per_chunk>0, "write_csv: invalid chunk_rows; must be at least 8" );

  auto exec = rmm::exec_policy(stream);
  
  auto num_rows = table.num_rows();
  std::vector<size_type> splits;

  if (num_rows <= n_rows_per_chunk )
    splits.push_back(num_rows);
  else {
    auto n_chunks = num_rows / n_rows_per_chunk;
    splits.resize(n_chunks);

    rmm::device_vector<size_type> d_splits(n_chunks, n_rows_per_chunk);
    thrust::inclusive_scan(exec->on(stream),
                           d_splits.begin(), d_splits.end(),
                           d_splits.begin());

    CUDA_TRY(cudaMemcpyAsync(splits.data(), d_splits.data().get(), 
                             n_chunks*sizeof(size_type), cudaMemcpyDeviceToHost,
                             stream));

    CUDA_TRY(cudaStreamSynchronize(stream));
  }

  //split table_view into chunks:
  //
  auto vector_views = cudf::experimental::split(table, splits);

  //convert each chunk to CSV:
  //
  for(auto&& sub_view: vector_views) {
    std::vector<std::unique_ptr<column>> str_column_vec;
    column_to_strings_fn converter{options_, mr_};

    //populate vector of string-converted columns:
    //
    std::transform(sub_view.begin(), sub_view.end(),
                   std::back_inserter(str_column_vec),
                   [converter](auto const& current_col) {
                     return cudf::experimental::type_dispatcher(current_col.type(),
                                                                converter,
                                                                current_col);
                   });

    //create string table view from str_column_vec:
    //
    auto str_table_ptr = std::make_unique<cudf::experimental::table>(std::move(str_column_vec));
    table_view str_table_view{std::move(*str_table_ptr)};

    //concatenate columns in each row into one big string column
    //(using null representation and delimiter):
    //
    std::string delimiter_str{options_.inter_column_delimiter()};
    auto str_concat_col = cudf::strings::concatenate(str_table_view,
                                                     delimiter_str,
                                                     options_.na_rep(),
                                                     mr_);

    strings_column_view strings_converted{std::move(*str_concat_col)};
    write_chunked(strings_converted, metadata, stream);
  }

  //finalize (no-op, for now, but offers a hook for future extensions):
  //
  write_chunked_end(table, metadata, stream);
}

void writer::write_all(table_view const &table, const table_metadata *metadata, cudaStream_t stream) {
  _impl->write(table, metadata, stream);
}



}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf

