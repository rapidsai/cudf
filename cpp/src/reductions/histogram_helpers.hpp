/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <optional>

namespace cudf::reduction::detail {

/**
 * @brief Compute the histogram for the input table.
 *
 * This is equivalent to do a distinct count for each unique rows in the input.
 *
 * @param input The input table to compute histogram
 * @param partial_counts An optional column containing counts for each row
 * @param output_dtype The output type to store the count value
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate memory of the returned objects
 * @return A pair of array contains the indices of the distinct rows in the input table, and their
 *         corresponding distinct counts
 */
std::pair<rmm::device_uvector<size_type>, std::unique_ptr<column>> table_histogram(
  table_view const& input,
  std::optional<column_view> const& partial_counts,
  data_type const output_dtype,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr);

}  // namespace cudf::reduction::detail
