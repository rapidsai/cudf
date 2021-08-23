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

#include <cudf/types.hpp>

namespace cudf {
namespace detail {

// mean and weight column indices within tdigest struct columns
constexpr size_type tdigest_mean_column_index   = 0;
constexpr size_type tdigest_weight_column_index = 1;

/**
 * @brief Verifies that the input column is a valid tdigest column.
 *
 * A tdigest column has the following form
 * list {
 *  struct {
 *    double    // mean
 *    double    // weight
 *  }
 * }
 * or:  list<struct<double, double>>
 * Each row represents a unique tdigest.
 *
 * @param col    Column to be checkeed
 *
 * @throws cudf::logic error if the column is not a valid tdigest column.
 */
void check_is_valid_tdigest_column(column_view const& col);

}  // namespace detail
}  // namespace cudf