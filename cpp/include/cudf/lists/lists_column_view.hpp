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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

/**
 * @file
 * @brief Class definition for cudf::lists_column_view
 */

namespace cudf {

/**
 * @addtogroup lists_classes
 * @{
 */

/**
 * @brief Given a column-view of lists type, an instance of this class
 * provides a wrapper on this compound column for list operations.
 */
class lists_column_view : private column_view {
 public:
  lists_column_view(column_view const& lists_column);
  lists_column_view(lists_column_view&& lists_view)      = default;
  lists_column_view(const lists_column_view& lists_view) = default;
  ~lists_column_view()                                   = default;
  lists_column_view& operator=(lists_column_view const&) = default;
  lists_column_view& operator=(lists_column_view&&) = default;

  static constexpr size_type offsets_column_index{0};
  static constexpr size_type child_column_index{1};

  using column_view::child_begin;
  using column_view::child_end;
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
   * @brief Returns the internal child column
   *
   * @throw cudf::logic error if this is an empty column
   */
  column_view child() const;

  /**
   * @brief Returns the internal child column, applying any offset from the root.
   *
   * Slice/split offset values are only stored at the root level of a list column.
   * So when doing computations on them, we need to apply that offset to
   * the child columns when recursing.  Most functions operating in a recursive manner
   * on lists columns should be using `get_sliced_child()` instead of `child()`.
   *
   * @throw cudf::logic error if this is an empty column
   */
  column_view get_sliced_child(rmm::cuda_stream_view stream) const;
};
/** @} */  // end of group
}  // namespace cudf
