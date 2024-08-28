/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace CUDF_EXPORT cudf {
namespace strings::detail {

/**
 * @copydoc concatenate(table_view const&,string_scalar const&,string_scalar
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    string_scalar const& separator,
                                    string_scalar const& narep,
                                    separator_on_nulls separate_nulls,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc join_strings(table_view const&,string_scalar const&,string_scalar
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> join_strings(strings_column_view const& strings,
                                     string_scalar const& separator,
                                     string_scalar const& narep,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc join_list_elements(table_view const&,string_scalar const&,string_scalar
 * const&,separator_on_nulls,output_if_empty_list,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> join_list_elements(lists_column_view const& lists_strings_column,
                                           string_scalar const& separator,
                                           string_scalar const& narep,
                                           separator_on_nulls separate_nulls,
                                           output_if_empty_list empty_list_policy,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

}  // namespace strings::detail
}  // namespace CUDF_EXPORT cudf
