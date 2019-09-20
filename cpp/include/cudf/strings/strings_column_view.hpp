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

#include <cudf/column/column_view.hpp>
#include <cudf/column/column.hpp>

namespace cudf {

/**---------------------------------------------------------------------------*
 * @brief Given a column-view of strings type, an instance of this class
 * provides a wrapper on this compound column for strings operations.
 *---------------------------------------------------------------------------**/
class strings_column_view : private column_view
{
 public:
  strings_column_view( column_view strings_column );
  strings_column_view( strings_column_view&& strings_view ) = default;
  strings_column_view( const strings_column_view& strings_view ) = default;
  ~strings_column_view() = default;

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of strings in the column
   *---------------------------------------------------------------------------**/
  size_type size() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the internal parent string column
   *---------------------------------------------------------------------------**/
  column_view parent() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the internal column of offsets
   *---------------------------------------------------------------------------**/
  column_view offsets() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the internal column of chars
   *---------------------------------------------------------------------------**/
  column_view chars() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns a pointer to the internal null mask memory
   *---------------------------------------------------------------------------**/
  const bitmask_type* null_mask() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of nulls in this column
   *---------------------------------------------------------------------------**/
  size_type null_count() const;

private:
  const column_view _parent;

};

namespace strings
{

/**---------------------------------------------------------------------------*
 * @brief Prints the strings to stdout.
 *
 * @param strings Strings instance for this operation.
 * @param start Index of first string to print.
 * @param end Index of last string to print. Specify -1 for all strings.
 * @param max_width Maximum number of characters to print per string.
 *        Specify -1 to print all characters.
 * @param delimiter The chars to print between each string.
 *        Default is new-line character.
 *---------------------------------------------------------------------------**/
void print( strings_column_view strings,
            size_type start=0, size_type end=-1,
            size_type max_width=-1, const char* delimiter = "\n" );


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
 * @param strings Strings instance for this operation.
 * @param start Index of first string to use.
 * @param end Index of last string to use.
 *        Default -1 indicates the last element.
 * @param step Increment value between indexes.
 *        Default step is 1.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column of size (end-start)/step.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> sublist( strings_column_view strings,
                                       size_type start, size_type end=-1,
                                       size_type step=1,
                                       cudaStream_t stream=0,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

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
 * @param strings Strings instance for this operation.
 * @param gather_map The indices with which to select strings for the new column.
 *        Values must be within [0,size()) range.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column of size indices.size()
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> gather( strings_column_view strings,
                                      cudf::column_view gather_map,
                                      cudaStream_t stream=0,
                                      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

// sort types can be combined
enum sort_type {
    none=0,    ///< no sorting
    length=1,  ///< sort by string length
    name=2     ///< sort by characters code-points
};

/**---------------------------------------------------------------------------*
 * @brief Returns a new strings column that is a sorted version of the
 * strings in this instance.
 *
 * @param strings Strings instance for this operation.
 * @param stype Specify what attribute of the string to sort on.
 * @param ascending Sort strings in ascending or descending order.
 * @param nullfirst Sort nulls to the beginning or the end of the new column.
 * @param stream CUDA stream to use kernels in this method.
 * @return New strings column with sorted elements of this instance.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> sort( strings_column_view strings,
                                    sort_type stype,
                                    bool ascending=true,
                                    bool nullfirst=true,
                                    cudaStream_t stream=0,
                                    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

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
 * @param strings Strings instance for this operation.
 * @param values The instance for which to retrieve the strings
 *        specified in map column.
 * @param scatter_map The 0-based index values to retrieve from the
 *        strings parameter. Number of values must equal the number
 *        of elements in strings pararameter (strings.size()).
 * @param stream CUDA stream to use kernels in this method.
 * @return New instance with the specified strings.
 */
std::unique_ptr<cudf::column> scatter( strings_column_view strings,
                                       strings_column_view values,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream=0,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );
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
 * @param strings Strings instance for this operation.
 * @param value Null-terminated encoded string in host memory to use with
 *        the scatter_map.
 * @param scatter_map The 0-based index values to place the given string.
 * @return New instance with the specified strings.
 */
std::unique_ptr<cudf::column> scatter( strings_column_view strings,
                                       const char* value,
                                       cudf::column_view scatter_map,
                                       cudaStream_t stream=0,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

// attributes.cu
/**---------------------------------------------------------------------------*
 * @brief Returns the number of bytes for each string in a strings column.
 * Null strings will have a byte count of 0.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream to use kernels in this method.
 * @return Numeric column of type int32.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> bytes_counts( strings_column_view strings,
                                            cudaStream_t stream=0,
                                            rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Returns the number of characters for each string in a strings column.
 * Null strings will have a count of 0. The number of characters is not the
 * same as the number of bytes if multi-byte encoded characters make up a
 * string.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream to use kernels in this method.
 * @return Numeric column of type int32.
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> characters_counts( strings_column_view strings,
                                                 cudaStream_t stream=0,
                                                 rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

/**---------------------------------------------------------------------------*
 * @brief Creates a column with code point values (integers) for each string.
 * A code point is the integer value representation of a character.
 * For example, in UTF-8 the code point value for the character 'A' is 65.
 * The column is an array of variable-length integer arrays each with length
 * as returned by characters_counts().
 * 
 * @code
 * s = ["a","xyz", "éee"]
 * v = code_points(s)
 * v is [97, 120, 121, 122, 50089, 101, 101]
 * @endcode
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream to use kernels in this method.
 * @return Numeric column of type int32. TODO: need uint32 here
 *---------------------------------------------------------------------------**/
std::unique_ptr<cudf::column> code_points( strings_column_view strings,
                                           cudaStream_t stream=0,
                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );

} // namespace strings
} // namespace cudf
