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

#ifndef GROUPBY_HPP
#define GROUPBY_HPP

#include <vector>


namespace cudf {

class table;

namespace groupby {

enum distributive_operators { SUM, MIN, MAX, COUNT };

/**---------------------------------------------------------------------------*
 * @brief Performs distributive groupby operation(s).
 *
 * Given a table of keys and corresponding table of values, equivalent keys will
 * be grouped together and a reduction operation performed across the associated
 * values (i.e., reduce by key). The reduction operation to be performed is
 * specified by a list of operator enums of equal length to the number of value
 * columns.
 *
 * The output of the operation is the table of key columns that hold all the
 * unique keys from the input key columns and a table of aggregation columns
 * that hold the specified reduction among all identical keys.
 *
 * @param keys The table of keys
 * @param values The table of aggregation values
 * @param operators The list of distributive aggregation operations
 * @return A tuple whose first member contains the table of output keys, and
 * second member contains the table of reduced output values
 *---------------------------------------------------------------------------**/
std::tuple<cudf::table, cudf::table> distributive(
    cudf::table const& keys, cudf::table const& values,
    std::vector<distributive_operators> const& operators);

}  // namespace groupby
}  // namespace cudf

#endif
