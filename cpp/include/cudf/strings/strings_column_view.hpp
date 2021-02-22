/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_vector.hpp>

/**
 * @file
 * @brief Class definition for cudf::strings_column_view
 */

namespace cudf {

/**
 * @addtogroup strings_classes
 * @{
 */

/**
 * @brief Given a column-view of strings type, an instance of this class
 * provides a wrapper on this compound column for strings operations.
 */
class strings_column_view : private column_view {
 public:
  strings_column_view(column_view strings_column);
  strings_column_view(strings_column_view&& strings_view)      = default;
  strings_column_view(const strings_column_view& strings_view) = default;
  ~strings_column_view()                                       = default;
  strings_column_view& operator=(strings_column_view const&) = default;
  strings_column_view& operator=(strings_column_view&&) = default;

  static constexpr size_type offsets_column_index{0};
  static constexpr size_type chars_column_index{1};

  using column_view::has_nulls;
  using column_view::is_empty;
  using column_view::null_count;
  using column_view::null_mask;
  using column_view::offset;
  using column_view::size;

  /**
   * @brief Returns the parent column.
   */
  column_view parent() const;

  /**
   * @brief Returns the internal column of offsets
   *
   * @throw cudf::logic error if this is an empty column
   */
  column_view offsets() const;

  /**
   * @brief Returns the internal column of chars
   *
   * @throw cudf::logic error if this is an empty column
   */
  column_view chars() const;

  /**
   * @brief Returns the number of bytes in the chars child column.
   *
   * This accounts for empty columns but does not reflect a sliced parent column
   * view  (i.e.: non-zero offset or reduced row count).
   */
  size_type chars_size() const noexcept;
};

//! Strings column APIs.
namespace strings {
/**
 * @brief Prints the strings to stdout.
 *
 * @param strings Strings instance for this operation.
 * @param start Index of first string to print.
 * @param end Index of last string to print. Specify -1 for all strings.
 * @param max_width Maximum number of characters to print per string.
 *        Specify -1 to print all characters.
 * @param delimiter The chars to print between each string.
 *        Default is new-line character.
 */
void print(strings_column_view const& strings,
           size_type start       = 0,
           size_type end         = -1,
           size_type max_width   = -1,
           const char* delimiter = "\n");

/**
 * @brief Create output per Arrow strings format.
 *
 * The return pair is the vector of chars and the vector of offsets.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned device_vectors.
 * @return Pair containing a vector of chars and a vector of offsets.
 */
std::pair<rmm::device_vector<char>, rmm::device_vector<size_type>> create_offsets(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace strings
/** @} */  // end of group
}  // namespace cudf
