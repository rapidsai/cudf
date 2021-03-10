/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

// TODO: Clean up includes when all debugging is done.
#include <cudf/binning/bin.hpp>
#include <cudf/binning/bin.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/types.hpp>

namespace cudf {

namespace bin {

// Bin the input by the edges in left_edges and right_edges.
std::unique_ptr<column> bin(column_view const& input, 
                            column_view const& left_edges,
                            inclusive left_inclusive,
                            column_view const& right_edges,
                            inclusive right_inclusive,
                            rmm::mr::device_memory_resource * mr)
{
    // TODO: Check if we want an empty set of bins to be an error, or if it should just return NULL for all bins.
    CUDF_EXPECTS(input.type() == left_edges.type(), "The input and edge columns must have the same types.");
    CUDF_EXPECTS(input.type() == right_edges.type(), "The input and edge columns must have the same types.");
    CUDF_EXPECTS(left_edges.size() == right_edges.size(), "The left and right edge columns must be of the same length.");

    // Handle empty inputs.
    if (input.is_empty()) {
        return cudf::make_numeric_column(data_type(type_to_id<size_type>()), 0);
    }

    return type_dispatcher(
            input.type(), bin_type_dispatcher{}, input, left_edges, left_inclusive, right_edges, right_inclusive, mr);
}
}  // namespace bin
}  // namespace cudf
