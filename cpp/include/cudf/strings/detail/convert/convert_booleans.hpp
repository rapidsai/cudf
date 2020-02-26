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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_booleans.hpp>

namespace cudf
{
namespace strings
{
namespace detail
{


/**
 * @copydoc cudf::strings::to_booleans
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */

std::unique_ptr<column> to_booleans( strings_column_view const& strings,
                                     string_scalar const& true_string = string_scalar("true"),
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream = 0);

/**
 * @copydoc cudf::strings::from_booleans
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::unique_ptr<column> from_booleans( column_view const& booleans,
                                       string_scalar const& true_string = string_scalar("true"),
                                       string_scalar const& false_string = string_scalar("false"),
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                       cudaStream_t stream = 0);
} // namespace detail
} // namespace strings
} // namespace cudf
