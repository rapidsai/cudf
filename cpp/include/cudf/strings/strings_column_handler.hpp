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

#include <vector>

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

  strings_column_handler( const column_view& strings_column, rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource() );
  //strings_column_handler( const column_view&& strings_column );

  /**---------------------------------------------------------------------------*
   * @brief Returns the number of strings in the column
   *---------------------------------------------------------------------------**/
  size_type count() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns a pointer to the internal char data array
   *---------------------------------------------------------------------------**/
  const char* chars_data() const;
  /**---------------------------------------------------------------------------*
   * @brief Returns a pointer to the internal offsets array
   *---------------------------------------------------------------------------**/
  const int32_t* offsets_data() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns the size of the char data array in bytes
   *---------------------------------------------------------------------------**/
  size_type chars_column_size() const;

  /**---------------------------------------------------------------------------*
   * @brief Returns a pointer to the internal null mask memory
   *---------------------------------------------------------------------------**/
  const bitmask_type* null_mask() const;
  /**---------------------------------------------------------------------------*
   * @brief Returns the number of nulls in this column
   *---------------------------------------------------------------------------**/
  size_type null_count() const;

  enum sort_type {
        none=0,    ///< no sorting
        length=1,  ///< sort by string length
        name=2     ///< sort by characters code-points
    };

  /**---------------------------------------------------------------------------*
   * @brief Prints the strings to stdout.
   * 
   * @param start Index of first string to print.
   * @param end Index of last string to print. Specify -1 for all strings.
   * @param max_width Maximum number of characters to print per string.
   *                  Specify -1 to print all characters.
   * @param delimiter The chars to print between each string.
   *                  Default is new-line character.
   *---------------------------------------------------------------------------**/
  void print( size_type start=0, size_type end=-1,
              size_type max_width=-1, const char* delimiter = "\n" ) const;

  /**---------------------------------------------------------------------------*
   * @brief Returns a new strings column created from a subset of
   * of this instance's strings column.
   * 
   * @param start Index of first string to use.
   * @param end Index of last string to use.
   * @param step Increment value between indexes.
   * @param stream CUDA stream to use kernels in this method.
   * @return New strings column of size (end-start)/step.
   *---------------------------------------------------------------------------**/
  std::unique_ptr<cudf::column> sublist( size_type start, size_type end, size_type step, cudaStream_t stream=0 );

  /**---------------------------------------------------------------------------*
   * @brief Returns a new strings column created this strings instance using
   * the specified indices to select the strings.
   * 
   * @param indices The indices with which to select strings for the new column.
   *                Values must be within [0,count()) range.
   * @param stream CUDA stream to use kernels in this method.
   * @return New strings column of size indices.size()
   *---------------------------------------------------------------------------**/
  std::unique_ptr<cudf::column> gather( const column_view& indices, cudaStream_t stream=0 );

  // return sorted version of the given strings column
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
  std::unique_ptr<cudf::column> sort( sort_type stype, bool ascending=true, bool nullfirst=true, cudaStream_t stream=0 );

private:
  const column_view _parent;
  rmm::mr::device_memory_resource* _mr;
  
};

}
