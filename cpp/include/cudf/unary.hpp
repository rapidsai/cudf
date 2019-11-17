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

#include <memory>
#include <cudf/types.hpp>
#include <rmm/mr/default_memory_resource.hpp>

namespace cudf {
namespace experimental {

/**
 * @brief Creates a column of `BOOL8` elements where for every element in `input` `true`
 * indicates the value is null and `false` indicates the value is valid.
 *
 * @param[in] input A `column_view` as input
 *
 * @returns std::unique_ptr<cudf::column> A non-nulalble column of `BOOL8` elements with `true` representing `null` values.
 */
std::unique_ptr<cudf::column> is_null(cudf::column_view const& input);

/**
 * @brief Creates a column of `BOOL8` elements where for every element in `input` `true`
 * indicates the value is valid and `false` indicates the value is null.
 *
 * @param[in] input A `column_view` as input
 *
 * @returns std::unique_ptr<cudf::column> A non-nulalble column of `BOOL8` elements with `false` representing `null` values.
 */
std::unique_ptr<cudf::column> is_valid(cudf::column_view const& input);

/**
 * @brief  Casts data from dtype specified in input to dtype specified in output
 *
 * @param column_view Input column
 * @param out_type Desired datatype of output column
 *
 * @returns unique_ptr<column> Result of the cast operation
 * @throw cudf::logic_error if `out_type` is not a fixed-width type
 */
std::unique_ptr<column> cast(column_view const& input, data_type out_type,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

} // namespace experimental
} // namespace cudf
