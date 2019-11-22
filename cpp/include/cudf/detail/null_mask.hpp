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

namespace cudf {

namespace detail {

/**---------------------------------------------------------------------------*
 * @brief Concatenates `views[i]`'s bitmask from the bits
 * `[views[i].offset(), views[i].offset() + views[i].size())` for all elements
 * views[i] in views into an array
 *
 * @param views Vector of column views whose bitmask needs to be copied
 * @param dest_mask Pointer to array that contains the combined bitmask
 * of the column views
 * @param stream stream on which all memory allocations and copies
 * will be performed
 *---------------------------------------------------------------------------**/
void concatenate_masks(std::vector<column_view> const &views,
    bitmask_type * dest_mask,
    cudaStream_t stream);

}  // namespace detail

}  // namespace cudf
