i/*
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

#include <cudf/cudf.h>
#include <cudf/types.hpp>

namespace cudf {
namespace experimental {
namespace detail{

/**
 * @brief Checks the `input` column_view for `null` values, and creates a `bool`
 * column of same size with `null`s being represented as `false` and others as `true`
 * if `nulls_are_false` is `true`, else `null`s will be represented by `true` and
 * others as `false`.
 *
 * @param[in] input A `column_view` as input
 * @param[in] nulls_are_false Value to represent `null`s
 * @param[in] mr Optional, The resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to execute kernels
 *
 * @returns std::unique_ptr<cudf::column> A column of type `BOOL8,` with `true` representing `null` values.
 */
std::unique_ptr<column> null_op(column_view const& input,
                                bool nulls_are_false = true,
                                rmm::mr::device_memory_resource* mr =
                                  rmm::mr::get_default_resource(),
                                cudaStream_t stream = 0)
} // namespace detail
} //namespace experimental
} //namespace cudf
