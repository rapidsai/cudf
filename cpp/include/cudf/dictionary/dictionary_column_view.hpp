/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

/**
 * @file
 * @brief Class definition for cudf::dictionary_column_view
 */

namespace CUDF_EXPORT cudf {
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
  /**
   * @brief Construct a new dictionary column view object from a column view.
   *
   * @param dictionary_column The column view to wrap
   */
  dictionary_column_view(column_view const& dictionary_column);
  dictionary_column_view(dictionary_column_view&&)      = default;  ///< Move constructor
  dictionary_column_view(dictionary_column_view const&) = default;  ///< Copy constructor
  ~dictionary_column_view() override                    = default;

  /**
   * @brief Move assignment operator
   *
   * @return The reference to this dictionary column
   */
  dictionary_column_view& operator=(dictionary_column_view const&) = default;

  /**
   * @brief Copy assignment operator
   *
   * @return The reference to this dictionary column
   */
  dictionary_column_view& operator=(dictionary_column_view&&) = default;

  /// Index of the indices column of the dictionary column
  static constexpr size_type indices_column_index{0};
  /// Index of the keys column of the dictionary column
  static constexpr size_type keys_column_index{1};

  using column_view::has_nulls;
  using column_view::is_empty;
  using column_view::null_count;
  using column_view::null_mask;
  using column_view::offset;
  using column_view::size;

  /**
   * @brief Returns the parent column.
   *
   * @return The parent column
   */
  [[nodiscard]] column_view parent() const noexcept;

  /**
   * @brief Returns the column of indices
   *
   * @return The indices column
   */
  [[nodiscard]] column_view indices() const noexcept;

  /**
   * @brief Returns a column_view combining the indices data
   * with offset, size, and nulls from the parent.
   *
   * @return A sliced indices column view with nulls from the parent
   */
  [[nodiscard]] column_view get_indices_annotated() const noexcept;

  /**
   * @brief Returns the column of keys
   *
   * @return  The keys column
   */
  [[nodiscard]] column_view keys() const noexcept;

  /**
   * @brief Returns the cudf::data_type of the keys child column.
   *
   * @return The cudf::data_type of the keys child column
   */
  [[nodiscard]] data_type keys_type() const noexcept;

  /**
   * @brief Returns the number of rows in the keys column.
   *
   * @return The number of rows in the keys column
   */
  [[nodiscard]] size_type keys_size() const noexcept;
};
/** @} */  // end of group

//! Dictionary column APIs.
namespace dictionary {  // defined here for doxygen output
}

}  // namespace CUDF_EXPORT cudf
