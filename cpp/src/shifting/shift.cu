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
 
#include <cudf/cudf.h>
#include <cudf/legacy/table.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>

#include <copying/copy_range.cuh>
#include <filling/fill.cuh>

#include <vector>

#include "shift.cuh"

namespace cudf {

void shift(
    table *out_table,
    table const &in_table,
    gdf_index_type period,
    gdf_scalar const &fill_value
)
{
    // TODO(cwharris): assert in / out same row count
    auto num_rows = out_table->num_rows();

    for (gdf_index_type i = 0; i < out_table->num_columns(); i++)
    {
        auto out_column = const_cast<gdf_column*>(out_table->get_column(i));
        auto in_column = const_cast<gdf_column*>(in_table.get_column(i));

        detail::shift(out_column, *in_column, period, fill_value);
    }
}

}; // namespace cudf
