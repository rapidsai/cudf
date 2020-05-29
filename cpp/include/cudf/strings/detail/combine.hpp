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

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @copydoc concatenate(table_view const&,string_scalar const&,string_scalar
 * const&,rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    string_scalar const& separator,
                                    string_scalar const& narep,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource* mr);

/**
 * @copydoc join_strings(table_view const&,string_scalar const&,string_scalar
 * const&,rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> join_strings(strings_column_view const& strings,
                                     string_scalar const& separator,
                                     string_scalar const& narep,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
