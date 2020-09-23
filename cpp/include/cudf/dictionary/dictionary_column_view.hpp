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

/**
 * @file
 * @brief Class definition for cudf::dictionary_column_view
 */

namespace cudf {
/**
 * @addtogroup dictionary_classes
 * @{
 */

/**
 * @brief A wrapper class for operations on a dictionary column.
 *
 * A dictionary column contains a set of keys and a column of indices.
 * The keys are a sorted set of unique values for the column.
 * The indices represent the corresponding positions of each element's
 * value in the keys.
 */
class dictionary_column_view : private column_view {
 public:
  dictionary_column_view(column_view const& dictionary_column);
  dictionary_column_view(dictionary_column_view&& dictionary_view)      = default;
  dictionary_column_view(const dictionary_column_view& dictionary_view) = default;
  ~dictionary_column_view()                                             = default;
  dictionary_column_view& operator=(dictionary_column_view const&) = default;
  dictionary_column_view& operator=(dictionary_column_view&&) = default;

  using column_view::has_nulls;
  using column_view::null_count;
  using column_view::null_mask;
  using column_view::offset;
  using column_view::size;

  /**
   * @brief Returns the parent column.
   */
  column_view parent() const noexcept;

  /**
   * @brief Returns the column of indices
   */
  column_view indices() const noexcept;

  /**
   * @brief Returns a column_view combining the indices data
   * with offset, size, and nulls from the parent.
   */
  column_view get_indices_annotated() const noexcept;

  /**
   * @brief Returns the column of keys
   */
  column_view keys() const noexcept;

  /**
   * @brief Returns the number of rows in the keys column.
   */
  size_type keys_size() const noexcept;
};
/** @} */  // end of group

//! Dictionary column APIs.
namespace dictionary {  // defined here for doxygen output
}

}  // namespace cudf
