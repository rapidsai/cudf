/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/table/table_view.hpp>

/**
 * @brief Estimates the column size in bytes.
 * 
 * @remark As this function internally uses cudf::row_bit_count() to estimate each row size
 * and accumulates them, the returned estimate may be an inexact approximation in some
 * cases. See cudf::row_bit_count() for more details.
 * 
 * @param view The column view to estimate its size
 */
int64_t estimate_size(cudf::column_view const& view);

/**
 * @brief Estimates the table size in bytes.
 * 
 * @remark As this function internally uses cudf::row_bit_count() to estimate each row size
 * and accumulates them, the returned estimate may be an inexact approximation in some
 * cases. See cudf::row_bit_count() for more details.
 * 
 * @param view The table view to estimate its size
 */
int64_t estimate_size(cudf::table_view const& view);
