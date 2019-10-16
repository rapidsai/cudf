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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf {

/**
 * @brief Returns a new table transposed from the input table
 * 
 * @param[in] input Input table of (ncols) number of columns each of size (nrows)
 * @return std::unique_ptr<table> Newly allocated output table
 */
std::unique_ptr<table> transpose(cudf::table_view input,
                                 device_memory_resource* mr = rmm::get_default_resource()));

}  // namespace cudf
