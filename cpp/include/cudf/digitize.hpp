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

#include <cudf/column/column_view.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/types.hpp>

namespace cudf {

/**
 * @brief Finds the indices of the bins in which each value of the column
 * belongs.
 *
 * For `x` in `col`, if `right == false` this function finds
 * `i` such that `bins[i-1] <= x < bins[i]`. If `right == true`, it will find `i`
 * such that `bins[i - 1] < x <= bins[i]`. Finally, if `x < bins[0]` or
 * `x > bins[num_bins - 1]`, it sets the index to `0` or `num_bins`, respectively.
 *
 * NOTE: This function does not handle null values and will throw if `col`
 * or `bins` contain any.
 * 
 * @throws cudf::logic_error if `col` and `bins` types mismatch
 * @throws cudf::logic_error if type is not numeric or timestamp
 * @throws cudf::logic_error if `col` or `bins` contain any null values
 *
 * @param col column_view with the values to be binned
 * @param bins column_view of ascending bin boundaries
 * @param right Whether the intervals should include the left or right bin edge
 * @param mr Optional resource to use for device memory allocation
 *
 * @returns device array of same size as `col` to be filled with bin indices
 */
rmm::device_vector<size_type>
digitize(column_view const& col, column_view const& bins, bool right,
         rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cudf
