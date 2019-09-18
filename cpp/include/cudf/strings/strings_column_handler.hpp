/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#pragma once

#include <rmm/thrust_rmm_allocator.h>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column.hpp>

namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Given a column-view of strings type, an instance of this class
 * provides the strings operations on the column.
 *---------------------------------------------------------------------------**/
class strings_column_handler
{
 public:
  ~strings_column_handler() = default;

  strings_column_handler( column_view strings_column,
                          rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );
  //strings_column_handler( const column_view&& strings_column );

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of strings in the column
   *---------------------------------------------------------------------------**/
  size_type size() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the internal parent string column
   *---------------------------------------------------------------------------**/
  column_view parent_column() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the internal column of offsets
   *---------------------------------------------------------------------------**/
  column_view offsets_column() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the internal column of chars
   *---------------------------------------------------------------------------**/
  column_view chars_column() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns a pointer to the internal null mask memory
   *---------------------------------------------------------------------------**/
  const bitmask_type* null_mask() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of nulls in this column
   *---------------------------------------------------------------------------**/
  size_type null_count() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the registered memory resource
   *---------------------------------------------------------------------------**/
  rmm::mr::device_memory_resource* memory_resource() const;

  /**---------------------------------------------------------------------------*
   * @brief Prints the strings to stdout.
   *
   * @param start Index of first string to print.
   * @param end Index of last string to print. Specify -1 for all strings.
   * @param max_width Maximum number of characters to print per string.
   *        Specify -1 to print all characters.
   * @param delimiter The chars to print between each string.
   *        Default is new-line character.
   *---------------------------------------------------------------------------**/
  void print( size_type start=0, size_type end=-1,
              size_type max_width=-1, const char* delimiter = "\n" ) const;

  // sort types can be combined
  enum sort_type {
      none=0,    ///< no sorting
      length=1,  ///< sort by string length
      name=2     ///< sort by characters code-points
  };

private:
  const column_view _parent;
  rmm::mr::device_memory_resource* _mr;

};

namespace strings
{

// array.cu
/**---------------------------------------------------------------------------*
 * @brief Returns a new strings column created from a subset of
 * of this instance's strings column.
 *
 * @code
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * s2 = sublist( s1, 2 )
 * s2 is ["c", "d", "e", "f"]
 * @endcode
 * 
 * @param start Index of first string to use.
 * @param end Index of last string to use.
 *        Default -1 indicates the last element.
 * @param step Increment value between indexes.
 *        Default step is 1.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column of size (end-start)/step.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> sublist( strings_column_handler handler,
                                       size_type start, size_type end=-1,
                                       size_type step=1,
                                       cudaStream_t stream=(cudaStream_t)0 );

/**---------------------------------------------------------------------------*
 * @brief Returns a new strings column created this strings instance using
 * the specified indices to select the strings.
 * 
 * @code
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather( s1, map )
 * s2 is ["a", "c"]
 * @endcode
 *
 * @param gather_map The indices with which to select strings for the new column.
 *        Values must be within [0,size()) range.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column of size indices.size()
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> gather( strings_column_handler handler,
                                      cudf::column_view gather_map,
                                      cudaStream_t stream=(cudaStream_t)0 );

/**---------------------------------------------------------------------------*
 * @brief Returns a new strings column that is a sorted version of the
 * strings in this instance.
 *
 * @param stype Specify what attribute of the string to sort on.
 * @param ascending Sort strings in ascending or descending order.
 * @param nullfirst Sort nulls to the beginning or the end of the new column.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column with sorted elements of this instance.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> sort( strings_column_handler handler,
                                    strings_column_handler::sort_type stype,
                                    bool ascending=true,
                                    bool nullfirst=true,
                                    cudaStream_t stream=(cudaStream_t)0 );

/**
 * @brief Returns new instance using the provided map values and strings.
 * The map values specify the location in the new strings instance.
 * Missing values pass through from the handler instance into those positions.
 *
 * @code
 * s1 = ["a", "b", "c", "d"]
 * s2 = ["e", "f"]
 * map = [1, 3]
 * s3 = scatter( s1, s2, m1 )
 * s3 is ["a", "e", "c", "f"]
 * @endcode
 *
 * @param[in] strings The instance for which to retrieve the values
 *            specified in map column.
 * @param[in] scatter_map The 0-based index values to retrieve from the
 *            strings parameter. Number of values must equal the number
 *            of elements in strings pararameter (strings.size()).
 * @param stream CUDA stream to use kernels in this method.
 * @return New instance with the specified strings.
 */
std::unique_ptr<cudf::column> scatter( strings_column_handler handler,
                                       strings_column_handler strings,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream=(cudaStream_t)0 );
/**
 * @brief Returns new instance using the provided index values and a
 * single string. The map values specify where to place the string
 * in the new strings instance. Missing values pass through from
 * the handler instance at those positions.
 *
 * @code
 * s1 = ["a", "b", "c", "d"]
 * map = [1, 3]
 * s2 = scatter( s1, "e", m1 )
 * s2 is ["a", "e", "c", "e"]
 * @endcode
 * 
 * @param[in] string The string to place in according to the scatter_map.
 * @param[in] scatter_map The 0-based index values to place the given string.
 * @return New instance with the specified strings.
 */
std::unique_ptr<cudf::column> scatter( strings_column_handler handler,
                                       const char* string,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream=(cudaStream_t)0 );

// attributes.cu
/**---------------------------------------------------------------------------*
 * @brief Returns the number of bytes for each string in a strings column.
 * Null strings will have a byte count of 0.
 *
 * @param stream CUDA stream to use kernels in this method.
 * @return Numeric column of type int32.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> bytes_counts( strings_column_handler handler,
                                            cudaStream_t stream=(cudaStream_t)0 );

/**---------------------------------------------------------------------------*
 * @brief Returns the number of characters for each string in a strings column.
 * Null strings will have a count of 0. The number of characters is not the
 * same as the number of bytes if multi-byte encoded characters make up a
 * string.
 *
 * @param stream CUDA stream to use kernels in this method.
 * @return Numeric column of type int32.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> characters_counts( strings_column_handler handler,
                                                 cudaStream_t stream=(cudaStream_t)0 );

} // namespace strings
} // namespace cudf
