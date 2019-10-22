/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include "cudf.h"
#include "types.hpp"

namespace cudf {
namespace experimental {

/**
 * @brief Checks the `input` column_view for `null` values, and creates a `bool`
 * column of same size with `true` representing `null` values and `false` for
 * other.
 *
 * @param input A `column_view` as input
 *
 * @returns std::unique_ptr<cudf::column> A column of type `BOOL8,` with `true` representing `null` values.
 */
std::unique_ptr<cudf::column> is_null(cudf::column_view const& input);

/**
 * @brief Checks the `input` column for `null` values, and creates a `bool`
 * column of same size with `false` representing `null` values and `true` for
 * other.
 *
 * @param input A `column_view` as input
 *
 * @returns std::unique_ptr<cudf::column>  A column of type `BOOL8` with `false` representing `null` values.
 */
std::unique_ptr<cudf::column> is_not_null(cudf::column_view const& input);

} // namespace experimental
} // namespace cudf
