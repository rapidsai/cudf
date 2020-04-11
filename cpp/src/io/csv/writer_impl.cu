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
#include <cudf/strings/strings_column_view.hpp>

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

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

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

  //TODO: `null` replacement with `na_rep` could be defered to `concatenate()`
  //instead of column-wise; might be faster
  //
  //TODO: pass `stream` to detail::<fname> version of <fname> calls below (?);

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

    strings_column_view strings_converted{std::move(*conv_col_ptr)};
    return  cudf::strings::replace_nulls(strings_converted,
                                         options_.na_rep() ,
                                         mr_);
  }

  //strings:
  //
  template<typename column_type>
  std::enable_if_t<std::is_same<column_type, cudf::string_view>::value,
                   std::unique_ptr<column>>
  operator()(column_view const& column) const
  {
    return cudf::strings::replace_nulls(column,
                                        options_.na_rep() ,
                                        mr_);
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

    strings_column_view strings_converted{std::move(*conv_col_ptr)};
    return cudf::strings::replace_nulls(strings_converted,
                                        options_.na_rep() ,
                                        mr_);
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

    strings_column_view strings_converted{std::move(*conv_col_ptr)};
    return cudf::strings::replace_nulls(strings_converted,
                                        options_.na_rep() ,
                                        mr_);
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

    strings_column_view strings_converted{std::move(*conv_col_ptr)};
    return cudf::strings::replace_nulls(strings_converted,
                                        options_.na_rep() ,
                                        mr_);
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


/**
 * @brief Helper function for write_csv.
 *
 * @param column The column to be converted.
 * @param options ...
 * @param mr...
 * @return strings_column_view instance formated for CSV column output.
**/
strings_column_view column_to_strings_csv(column_view const& column,
                                          writer_options const& options,
                                          rmm::mr::device_memory_resource* mr = nullptr) {
  //TODO;
  //
  column_to_strings_fn col2str{options, mr};
  auto ret = col2str.template operator()<bool>(column); // check instantiation: okay
  
  return strings_column_view{column}; // for now
}

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

void writer::impl::write_chunked_begin(table_view const& table,
                                       const table_metadata *metadata,
                                       cudaStream_t stream)
{
}


void writer::impl::write_chunked(table_view const& table,
                                 const table_metadata *metadata,
                                 cudaStream_t stream)
{
}

  
void writer::impl::write(table_view const &table,
                         const table_metadata *metadata,
                         cudaStream_t stream) {
  //TODO: chunked behavior / decision making (?)

  CUDF_EXPECTS( table.num_columns() > 0 && table.num_rows() > 0, "Empty table." );

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

    CUDA_TRY(cudaMemcpyAsync(d_splits.data().get(), splits.data(),
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

    std::string delimiter_str{options_.inter_column_delimiter()};
    auto str_concat_col = cudf::strings::concatenate(str_table_view,
                                                     delimiter_str,
                                                     options_.na_rep(),
                                                     mr_);
  //  thrust::copy(str_concat_col.begin(), str_concat_col.end(), data_sink_obj); 
  }
}

void writer::write_all(table_view const &table, const table_metadata *metadata, cudaStream_t stream) {
  _impl->write(table, metadata, stream);
}



}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf

