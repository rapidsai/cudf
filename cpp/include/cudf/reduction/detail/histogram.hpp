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

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>

namespace cudf::reduction::detail {

/**
 * @brief Compute the frequency for each distinct row in the input table.
 *
 * @param input The input table to compute histogram
 * @param partial_counts An optional column containing count for each row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate memory of the returned objects
 * @return A pair of array contains the (stable-order) indices of the distinct rows in the input
 * table, and their corresponding distinct counts
 */
[[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>, std::unique_ptr<column>>
compute_row_frequencies(table_view const& input,
                        std::optional<column_view> const& partial_counts,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr);

/**
 * @brief Create an empty histogram column.
 *
 * A histogram column is a structs column `STRUCT<T, int64_t>` where T is type of the input
 * values.
 *
 * @returns An empty histogram column
 */
[[nodiscard]] std::unique_ptr<column> make_empty_histogram_like(column_view const& values);

}  // namespace cudf::reduction::detail
