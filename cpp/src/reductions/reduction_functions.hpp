/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/scalar/scalar.hpp>

namespace cudf {
namespace experimental {
namespace reduction {

std::unique_ptr<scalar> sum(
    column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);
std::unique_ptr<scalar> min(
    column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);
std::unique_ptr<scalar> max(
    column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);
std::unique_ptr<scalar> any(
    column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);
std::unique_ptr<scalar> all(
    column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);
std::unique_ptr<scalar> product(
    column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);
std::unique_ptr<scalar> sum_of_squares(
    column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

std::unique_ptr<scalar> mean(
    column_view const& col, data_type const output_dtype,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);
std::unique_ptr<scalar> variance(
    column_view const& col, data_type const output_dtype, cudf::size_type ddof,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);
std::unique_ptr<scalar> standard_deviation(
    column_view const& col, data_type const output_dtype, cudf::size_type ddof,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

} // namespace reduction
} // namespace experimental
} // namespace cudf

