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
#include <cudf/strings/convert/convert_integers.hpp>

namespace cudf
{
namespace strings
{
namespace detail
{

/**
 * @copydoc cudf::strings::to_integers
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::unique_ptr<column> to_integers( strings_column_view const& strings,
                                     data_type output_type,
                                     rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                     cudaStream_t stream = 0);

/**
 * @copydoc cudf::strings::from_integers
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::unique_ptr<column> from_integers( column_view const& integers,
                                       rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                       cudaStream_t stream = 0);

/**
 * @copydoc cudf::strings::hex_to_integers
 *
 * @param[in] stream Optional CUDA stream on which to execute kernels
 */
std::unique_ptr<column> hex_to_integers( strings_column_view const& strings,
                                         data_type output_type,
                                         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                         cudaStream_t stream = 0);

} // namspace detail
} // namespace strings
} // namespace cudf
