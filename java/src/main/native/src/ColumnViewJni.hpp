/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

} // namespace cudf::jni
