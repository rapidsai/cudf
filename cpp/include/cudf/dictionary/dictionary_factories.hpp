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

#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/column/column.hpp>


namespace cudf
{

/**
 * @brief Construct a dictionary column from a column.
 *
 * Unique elements are managed in a dictionary.
 * The output column will have indices with the same count
 * as the input column.
 *
 * ```
 * c = [429,111,213,111,213,429,213]
 * d = make_dictionary_column(c)
 * d now has dictionary [111,213,429] and indices [2,0,1,0,1,2,1]
 * ```
 *
 * @param[in] column The column data to build the dictionary from.
 * @param[in] mr Optional resource to use for device memory allocation.
 * @param[in] stream Optional stream on which to issue all memory allocation and
 * device kernels.
 */
std::unique_ptr<column> make_dictionary_column(
    column_view const& column,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @brief Construct a dictionary column from a series of data.
 *
 * Unique elements are managed in a dictionary.
 * The output column will have indices with the same count
 * as the input column.
 *
 * @param[in] begin Start of data to retrieve. (inclusive)
 * @param[in] end End of data to retrieve. (exclusive)
 * @param null_count The number of null entries.
 * @param null_mask The bits specifying the null entries in device memory.
 * @param[in] mr Optional resource to use for device memory allocation.
 * @param[in] stream Optional stream on which to issue all memory allocation and
 * device kernels.
 */
template<typename Iterator>
std::unique_ptr<column> make_dictionary_column(
    Iterator begin, Iterator end,
    size_type null_count,
    rmm::device_buffer&& null_mask,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);


}  // namespace cudf
