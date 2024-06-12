/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <optional>

namespace cudf::reduction::detail {

/**
 * @copydoc cudf::reduce(column_view const&, reduce_aggregation const&, data_type,
 * std::optional<std::reference_wrapper<scalar const>>, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<scalar> reduce(column_view const& col,
                               reduce_aggregation const& agg,
                               data_type output_dtype,
                               std::optional<std::reference_wrapper<scalar const>> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

}  // namespace cudf::reduction::detail
