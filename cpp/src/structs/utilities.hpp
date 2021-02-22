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

#include <cudf/structs/structs_column_view.hpp>

namespace cudf {
namespace structs {
namespace detail {

/**
 * @brief Flatten the children of the input columns into a vector where the i'th element
 * is a vector of column_views representing the i'th child from each input column_view.
 *
 * @code{.pseudo}
 * s1 = [ col0 : {0, 1}
 *        col1 : {2, 3, 4, 5, 6}
 *        col2 : {"abc", "def", "ghi"} ]
 *
 * s2 = [ col0 : {7, 8}
 *        col1 : {-4, -5, -6}
 *        col2 : {"uvw", "xyz"} ]
 *
 * e = extract_ordered_struct_children({s1, s2})
 *
 * e is now [ {{0, 1}, {7, 8}}
 *            {{2, 3, 4, 5, 6}, {-4, -5, -6}}
 *            {{"abc", "def", "ghi"}, {"uvw", "xyz"} ]
 * @endcode
 *
 * @param columns Vector of structs columns to extract from.
 * @return New column with concatenated results.
 */
std::vector<std::vector<column_view>> extract_ordered_struct_children(
  std::vector<column_view> const& struct_cols);

}  // namespace detail
}  // namespace structs
}  // namespace cudf
