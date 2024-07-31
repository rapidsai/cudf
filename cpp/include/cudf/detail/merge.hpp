/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cudf/utilities/export.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/pair.h>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Source table identifier to copy data from.
 */
enum class side : bool { LEFT, RIGHT };

/**
 * @brief Tagged index type: `thrust::get<0>` indicates left/right side,
 * `thrust::get<1>` indicates the row index
 */
using index_type = thrust::pair<side, cudf::size_type>;

/**
 * @brief Vector of `index_type` values.
 */
using index_vector = rmm::device_uvector<index_type>;

/**
 * @copydoc std::unique_ptr<cudf::table> merge(
 *            std::vector<table_view> const& tables_to_merge,
 *            std::vector<cudf::size_type> const& key_cols,
 *            std::vector<cudf::order> const& column_order,
 *            std::vector<cudf::null_order> const& null_precedence,
 *            rmm::cuda_stream_view stream,
 *            rmm::device_async_resource_ref mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::table> merge(std::vector<table_view> const& tables_to_merge,
                                   std::vector<cudf::size_type> const& key_cols,
                                   std::vector<cudf::order> const& column_order,
                                   std::vector<cudf::null_order> const& null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
