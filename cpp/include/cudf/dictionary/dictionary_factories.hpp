/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>


namespace cudf
{

/**
 * @brief Construct a dictionary column by using the provided keys
 * and indices.
 *
 * The keys_column must contain no nulls.
 * It is assumed the elements in `keys_column` are unique and
 * are in a strict, total order. Meaning, `keys_column[i]` is _ordered before
 * `keys_column[i+1]` for all `i in [0,n-1)` where `n` is the number of keys.
 *
 * The indices values must be in the range [0,keys_column.size()).
 *
 * The null_mask and null count for the output column are copied from the indices column.
 *
 * ```
 * k = ["a","c","d"]
 * i = [1,0,0,2,2]
 * d = make_dictionary_column(k,i)
 * d is now {["a","c","d"],[1,0,0,2,2]}
 * ```
 *
 * @throw cudf::logic_error if keys_column contains nulls
 *
 * @param keys_column Column of unique, ordered values to use as the new dictionary column's keys.
 * @param indices_column Indices to use for the new dictionary column.
 * @param mr Resource for allocating memory for the output.
 * @param stream Optional stream on which to issue all memory allocation and
 *               device kernels.
 * @return New dictionary column.
 */
std::unique_ptr<column> make_dictionary_column( column_view const& keys_column,
                                                column_view const& indices_column,
                                                rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                                                cudaStream_t stream = 0);

}  // namespace cudf
