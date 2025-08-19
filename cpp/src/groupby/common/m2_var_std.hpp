/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace cudf::groupby::detail {

std::unique_ptr<column> compute_m2(data_type source_type,
                                   column_view const& sum_sqr,
                                   column_view const& sum,
                                   column_view const& count,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

std::unique_ptr<column> compute_variance(column_view const& m2,
                                         column_view const& count,
                                         size_type ddof,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

std::unique_ptr<column> compute_std(column_view const& m2,
                                    column_view const& count,
                                    size_type ddof,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

}  // namespace cudf::groupby::detail
