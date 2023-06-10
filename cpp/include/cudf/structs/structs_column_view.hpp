/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

/**
 * @file
 * @brief Class definition for cudf::structs_column_view.
 */

namespace cudf {

/**
 * @addtogroup structs_classes
 * @{
 */

/**
 * @brief Given a column view of struct type, an instance of this class
 * provides a wrapper on this compound column for struct operations.
 */
class structs_column_view : public column_view {
 public:
  // Foundation members:
  structs_column_view(structs_column_view const&) = default;  ///< Copy constructor
  structs_column_view(structs_column_view&&)      = default;  ///< Move constructor
  ~structs_column_view()                          = default;
  /**
   * @brief Copy assignment operator
   *
   * @return The reference to this structs column
   */
  structs_column_view& operator=(structs_column_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return The reference to this structs column
   */
  structs_column_view& operator=(structs_column_view&&) = default;

  /**
   * @brief Construct a new structs column view object from a column view.
   *
   * @param col The column view to wrap
   */
  explicit structs_column_view(column_view const& col);

  /**
   * @brief Returns the parent column.
   *
   * @return The parent column
   */
  [[nodiscard]] column_view parent() const;

  using column_view::child_begin;
  using column_view::child_end;
  using column_view::has_nulls;
  using column_view::null_count;
  using column_view::null_mask;
  using column_view::num_children;
  using column_view::offset;
  using column_view::size;

  /**
   * @brief Returns the internal child column, applying any offset from the root.
   *
   * Slice/split offset values are only stored at the root level of a struct column.
   * So when doing computations on them, we need to apply that offset to
   * the child columns when recursing.  Most functions operating in a recursive manner
   * on struct columns should be using `get_sliced_child()` instead of `child()`.
   *
   * @throw cudf::logic error if this is an empty column
   *
   * @param index The index of the child column to return
   * @param stream The stream on which to perform the operation. Uses the default CUDF
   *        stream if none is specified.
   * @return The child column sliced relative to the parent's offset and size
   */
  [[nodiscard]] column_view get_sliced_child(
    int index, rmm::cuda_stream_view stream = cudf::get_default_stream()) const;
};         // class structs_column_view;
/** @} */  // end of group
}  // namespace cudf
