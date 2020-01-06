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

#pragma once

#include <cudf/types.hpp>

#include <memory>

namespace cudf {
namespace experimental {

/**
 * @brief Creates a new column by applying a unary function against every
 * element of an input column.
 *
 * Computes:
 * `out[i] = F(in[i])`
 * 
 * The output null mask is the same is the input null mask so if input[i] is 
 * null then output[i] is also null
 *
 * @param input         An immutable view of the input column to transform
 * @param unary_udf     The PTX/CUDA string of the unary function to apply
 * @param outout_type   The output type that is compatible with the output type in the UDF
 * @param is_ptx        true: the UDF is treated as PTX code; false: the UDF is treated as CUDA code
 * @param mr            The memory resource to use for for all device allocations
 * @return cudf::column The column resulting from applying the unary function to
 *                      every element of the input
 **/
std::unique_ptr<column> transform(
  column_view const& input,
  std::string const& unary_udf,
  data_type output_type,
  bool is_ptx,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());


/**
 * @brief Creates a null_mask from `input` by converting `NaN` to null and
 * preserving existing null values and also returns new null_count.
 *
 * @throws `cudf::logic_error` if `input.type()` is a non-floating type
 *
 * @param input         An immutable view of the input column of floating-point type
 * @param mr            The memory resource to use for for all device allocations
 * @return A pair containing a `device_buffer` with the new bitmask and it's
 * null count obtained by replacing `NaN` in `input` with null.
 **/
std::pair<std::unique_ptr<rmm::device_buffer>, size_type>
nans_to_nulls(column_view const& input,
              rmm::mr::device_memory_resource * mr = rmm::mr::get_default_resource());
}  // namespace experimental
}  // namespace cudf
