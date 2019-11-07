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

#include <cudf/binaryop.hpp>

namespace cudf {
namespace experimental {
namespace detail {

// TODO: Document
std::unique_ptr<column> binary_operation( scalar const& lhs,
                                          column_view const& rhs,
                                          binary_operator ope,
                                          data_type output_type,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

std::unique_ptr<column> binary_operation( column_view const& lhs,
                                          scalar const& rhs,
                                          binary_operator ope,
                                          data_type output_type,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

std::unique_ptr<column> binary_operation( column_view const& lhs,
                                          column_view const& rhs,
                                          binary_operator ope,
                                          data_type output_type,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

std::unique_ptr<column> binary_operation( column_view const& lhs,
                                          column_view const& rhs,
                                          std::string const& ptx,
                                          data_type output_type,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

} // namespace detail
} // namespace experimental
} // namespace cudf
