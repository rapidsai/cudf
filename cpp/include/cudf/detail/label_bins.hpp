/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include <cudf/column/column_view.hpp>
#include <cudf/labeling/label_bins.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace CUDF_EXPORT cudf {

namespace detail {

/**
 * @addtogroup label_bins
 * @{
 * @file
 * @brief Internal APIs for labeling values by bin.
 */

/**
 * @copydoc cudf::label_bins(column_view const& input, column_view const& left_edges, inclusive
 * left_inclusive, column_view const& right_edges, inclusive right_inclusive, rmm::cuda_stream_view,
 * rmm::device_async_resource_ref mr)
 *
 * @param stream Stream view on which to allocate resources and queue execution.
 */
std::unique_ptr<column> label_bins(column_view const& input,
                                   column_view const& left_edges,
                                   inclusive left_inclusive,
                                   column_view const& right_edges,
                                   inclusive right_inclusive,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

/** @} */  // end of group
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
