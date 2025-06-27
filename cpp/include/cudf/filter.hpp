

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

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace CUDF_EXPORT cudf {

/**
 * @brief Creates a new column by applying a filter function against every
 * element of the input columns.
 * Null values in the input columns are considered as not matching the filter.
 *
 * Computes:
 * `out[i]... = predicate(columns[i]... ) ? (columns[i]...): not-applied`.
 *
 * Note that for every scalar in `predicate_columns` (columns of size 1), `predicate_columns[i] ==
 * input[0]`
 *
 *
 * @throws std::invalid_argument if any of the input columns have different sizes (except scalars of
 * size 1)
 * @throws std::invalid_argument if `output_type` or any of the inputs are not fixed-width or string
 * types
 * @throws std::invalid_argument if any of the input columns have nulls
 * @throws std::logic_error if JIT is not supported by the runtime
 * @throws std::logic_error if the size of the copy_mask does not match the size of the input
 *
 * The size of the resulting column is the size of the largest column.
 *
 * @param columns        Immutable views of the columns to filter
 * @param predicate_udf The PTX/CUDA string of the transform function to apply
 * @param is_ptx        true: the UDF is treated as PTX code; false: the UDF is treated as CUDA code
 * @param user_data     User-defined device data to pass to the UDF.
 * @param copy_mask  Optional vector of booleans indicating which columns to copy from the input
 *                   columns to the output. If not provided, all columns are copied.
 * @param stream        CUDA stream used for device memory operations and kernel launches
 * @param mr            Device memory resource used to allocate the returned column's device memory
 * @return              The filtered target columns
 */
std::vector<std::unique_ptr<column>> filter(
  std::vector<column_view> const& columns,
  std::string const& predicate_udf,
  bool is_ptx,
  std::optional<void*> user_data             = std::nullopt,
  std::optional<std::vector<bool>> copy_mask = std::nullopt,
  rmm::cuda_stream_view stream               = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr          = cudf::get_current_device_resource_ref());

}  // namespace CUDF_EXPORT cudf
