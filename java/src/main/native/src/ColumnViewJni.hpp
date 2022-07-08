/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace cudf::jni {

/**
 * @brief Creates a deep copy of the exemplar column, with its validity set to the equivalent
 * of the boolean `validity` column's value.
 *
 * The bool_column must have the same number of rows as the exemplar column.
 * The result column will have the same number of rows as the exemplar.
 * For all indices `i` where the boolean column is `true`, the result column will have a valid value
 * at index i. For all other values (i.e. `false` or `null`), the result column will have nulls.
 *
 * @param exemplar The column to be deep copied.
 * @param bool_column bool column whose value is to be used as the validity.
 * @return Deep copy of the exemplar, with the replaced validity.
 */
std::unique_ptr<cudf::column>
new_column_with_boolean_column_as_validity(cudf::column_view const &exemplar,
                                           cudf::column_view const &bool_column);

/**
 * @brief Generates list offsets with lengths of each list.
 *
 * For example,
 * Given a list column: [[1,2,3], [4,5], [6], [], [7,8]]
 * The list lengths of it: [3, 2, 1, 0, 2]
 * The list offsets of it: [0, 3, 5, 6, 6, 8]
 *
 * @param list_length The column represents list lengths.
 * @return The column represents list offsets.
 */
std::unique_ptr<cudf::column>
generate_list_offsets(cudf::column_view const &list_length,
                      rmm::cuda_stream_view stream = cudf::default_stream_value);

} // namespace cudf::jni
