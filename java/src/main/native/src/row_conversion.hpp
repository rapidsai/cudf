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

#include <memory>

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {
namespace java {


std::vector<std::unique_ptr<cudf::column>> convert_to_rows(
        cudf::table_view const& tbl,
        // TODO need something for validity
        cudaStream_t stream = 0,
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::table> convert_from_rows(
        cudf::lists_column_view const& input,
        std::vector<cudf::data_type> const& schema,
        cudaStream_t stream = 0,
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

} // java
} // cudf
