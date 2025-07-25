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

#include <cudf/io/types.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>

namespace cudf::examples {

/**
 * @brief Check if two tables are identical, throw an error otherwise
 *
 * @param lhs_table View to lhs table
 * @param rhs_table View to rhs table
 */
void check_tables_equal(cudf::table_view const& lhs_table, cudf::table_view const& rhs_table)
{
  try {
    // Left anti-join the original and transcoded tables
    // identical tables should not throw an exception and
    // return an empty indices vector
    auto const indices = cudf::left_anti_join(lhs_table, rhs_table, cudf::null_equality::EQUAL);

    // No exception thrown, check indices
    auto const valid = indices->size() == 0;
    std::cout << "Tables identical: " << std::boolalpha << valid << "\n\n";
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl << std::endl;
    throw std::runtime_error("Tables identical: false\n\n");
  }
}

}  // namespace cudf::examples
