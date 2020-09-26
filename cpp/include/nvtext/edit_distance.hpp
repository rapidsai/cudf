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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

//! NVText APIs
namespace nvtext {
/**
 * @addtogroup nvtext_edit_distance
 * @{
 * @file
 */

/**
 * @brief Compute the edit distance between individual strings in two strings columns.
 *
 * The `output[i]` is the edit distance between `strings[i]` and `targets[i]`.
 * This edit distance calculation uses the Levenshtein algorithm as documented here:
 * https://www.cuelogic.com/blog/the-levenshtein-algorithm
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "", "world"]
 * t = ["hallo", "goodbye", "world"]
 * d = edit_distance(s, t)
 * d is now [1, 7, 0]
 * @endcode
 *
 * Any null entries for either `strings` or `targets` is ignored and the edit distance
 * is computed as though the null entry is an empty string.
 *
 * The `targets.size()` must equal `strings.size()` unless `targets.size()==1`.
 * In this case, all `strings` will be computed against the single `targets[0]` string.
 *
 * @throw cudf::logic_error if `targets.size() != strings.size()` and
 *                          if `targets.size() != 1`
 *
 * @param strings Strings column of input strings
 * @param targets Strings to compute edit distance against `strings`
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings columns of with replaced strings.
 */
std::unique_ptr<cudf::column> edit_distance(
  cudf::strings_column_view const& strings,
  cudf::strings_column_view const& targets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute the edit distance between all the strings in the input column.
 *
 * This uses the Levenshtein algorithm to calculate the edit distance between
 * two strings as documented here: https://www.cuelogic.com/blog/the-levenshtein-algorithm
 *
 * The output is essentially a `strings.size() x strings.size()` square matrix of integers.
 * All values at diagonal `row == col` are 0 since the edit distance between two identical
 * strings is zero. All values above the diagonal are reflected below since the edit distance
 * calculation is also commutative.
 *
 * @code{.pseudo}
 * Example:
 * s = ["hello", "hallo", "hella"]
 * d = edit_distance_matrix(s)
 * d is now [[0, 1, 1],
 *           [1, 0, 2]
 *           [1, 2, 0]]
 * @endcode
 *
 * Null entries for `strings` are ignored and the edit distance
 * is computed as though the null entry is an empty string.
 *
 * The output is a lists column of size `strings.size()` and where each list item
 * is `strings.size()` elements.
 *
 * @throw cudf::logic_error if `strings.size() == 1`
 *
 * @param strings Strings column of input strings
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New lists column of edit distance values.
 */
std::unique_ptr<cudf::column> edit_distance_matrix(
  cudf::strings_column_view const& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace nvtext
