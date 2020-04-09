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

#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp> 

#include <algorithm>
#include <cstring>
#include <utility>
#include <type_traits>

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
  //collects the same conditions used for
  //instantiations of individual converters in strings/convert/convert_*.hpp
  //
  template<typename column_type>
  constexpr static bool is_not_handled(void)
  {
    return( (!std::is_same<column_type, bool>::value) &&
            (!std::is_same<column_type, cudf::string_view>::value) &&
            (!std::is_integral<column_type>::value) &&
            (!std::is_floating_point<column_type>::value) &&
            (!cudf::is_timestamp<column_type>()) );
  }
  
  column_to_strings_fn(writer_options const& options):
    options_(options)
  {
  }

  template<typename column_type>
  std::enable_if_t<std::is_same<column_type, bool>::value,
                   strings_column_view>
  operator()(column_view const& column) const
  {
    //bools...
    //TODO:
    //
    return strings_column_view(column);//for now
  }

  template<typename column_type>
  std::enable_if_t<std::is_same<column_type, cudf::string_view>::value,
                   strings_column_view>
  operator()(column_view const& column) const
  {
    //strings...
    return column;//strings_column_view(column);
  }

  template<typename column_type>
  std::enable_if_t<std::is_integral<column_type>::value,
                   strings_column_view>
  operator()(column_view const& column) const
  {
    //ints...
    //TODO:
    //
    return strings_column_view(column);//for now
  }

  template<typename column_type>
  std::enable_if_t<std::is_floating_point<column_type>::value,
                   strings_column_view>
  operator()(column_view const& column) const
  {
    //floats...
    //TODO:
    //
    return strings_column_view(column);//for now
  }

  template<typename column_type>
  std::enable_if_t<cudf::is_timestamp<column_type>(),
                   strings_column_view>
  operator()(column_view const& column) const
  {
    //timestamps...
    //TODO:
    //
    return strings_column_view(column);//for now
  }


  template<typename column_type>
  std::enable_if_t<is_not_handled<column_type>(),
                   strings_column_view>
  operator()(column_view const& column) const
  {
    //not to be called...
    //
    CUDF_FAIL("Unsupported column type.");
    //silence the compiler: no return
  }
private:
  writer_options const& options_;
};


/**
 * @brief Helper function for write_csv.
 *
 * @param column The column to be converted.
 * @param row_offset Number of entries from the beginning to skip; must be multiple of 8.
 * @param rows Number of rows from the offset that should be converted for this column.
 * @param delimiter Separator to append to the column strings
 * @param null_representation String to use for null entries
 * @param true_string String to use for 'true' values in boolean columns
 * @param false_string String to use for 'false' values in boolean columns
 * @return strings_column_view instance formated for CSV column output.
**/
strings_column_view column_to_strings_csv(column_view const& column,
                                          cudf::size_type row_offset,
                                          cudf::size_type rows,
                                          writer_options const& options) {
  //TODO;
  //
  column_to_strings_fn col2str{options};
  //col2str.template operator()<int>(column); // check instantiation: okay
  
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
  auto rows_chunk = options_.rows_per_chunk();
  //
  // This outputs the CSV in row chunks to save memory.
  // Maybe we can use the total_rows*count calculation and a memory threshold
  // instead of an arbitrary chunk count.
  // The entire CSV chunk must fit in CPU memory before writing it out.
  //
  if( rows_chunk % 8 ) // must be divisible by 8
    rows_chunk += 8 - (rows_chunk % 8);
  CUDF_EXPECTS( rows_chunk>0, "write_csv: invalid chunk_rows; must be at least 8" );

  auto exec = rmm::exec_policy(stream);
}

void writer::write_all(table_view const &table, const table_metadata *metadata, cudaStream_t stream) {
  _impl->write(table, metadata, stream);
}



}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf

