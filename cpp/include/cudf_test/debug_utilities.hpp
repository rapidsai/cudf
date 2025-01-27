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

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace test {

/**
 * @brief Formats a column view as a string
 *
 * @param col The input column view
 * @param delimiter The delimiter to put between strings
 */
std::string to_string(cudf::column_view const& col, std::string const& delimiter);

/**
 * @brief Convert column values to a host vector of strings
 *
 * @param col The input column view
 */
std::vector<std::string> to_strings(cudf::column_view const& col);

/**
 * @brief Print a column view to an ostream
 *
 * @param col The input column view
 * @param os The output stream
 */
void print(cudf::column_view const& col, std::ostream& os = std::cout);

}  // namespace test
}  // namespace CUDF_EXPORT cudf
