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

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <optional>
#include <utility>

namespace CUDF_EXPORT cudf {
namespace io::detail {

/**
 * @brief Adjusts the input skip_rows and num_rows options to the actual number of rows to
 * skip/read, based on the number of rows in the ORC file(s).
 *
 * @param skip_rows skip_rows as passed by the user
 * @param num_rows Optional num_rows as passed by the user
 * @param num_source_rows number of rows in the ORC file(s)
 * @return A std::pair containing the number of rows to skip and the number of rows to read
 *
 * @throw std::overflow_exception The requested number of rows exceeds the column size limit
 */
std::pair<int64_t, int64_t> skip_rows_num_rows_from_options(int64_t skip_rows,
                                                            std::optional<int64_t> const& num_rows,
                                                            int64_t num_source_rows);

}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
