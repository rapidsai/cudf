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

namespace cudf {
/**
 * @brief Construct a dictionary column by copying the provided `keys`
 * and `indices`.
 *
 * @ingroup column_factories
 *
 * It is expected that `keys_column.has_nulls() == false`.
 * It is assumed the elements in `keys_column` are unique and
 * are in a strict, total order. Meaning, `keys_column[i]` is ordered before
 * `keys_column[i+1]` for all `i in [0,n-1)` where `n` is the number of keys.
 *
 * The indices values must be in the range [0,keys_column.size()).
 *
 * The null_mask and null count for the output column are copied from the indices column.
 * If element `i` in `indices_column` is null, then element `i` in the returned dictionary column
 * will also be null.
 *
 * ```
 * k = ["a","c","d"]
 * i = [1,0,null,2,2]
 * d = make_dictionary_column(k,i)
 * d is now {["a","c","d"],[1,0,undefined,2,2]} bitmask={1,1,0,1,1}
 * ```
 *
 * The null_mask and null count for the output column are copied from the indices column.
 *
 * @throw cudf::logic_error if keys_column contains nulls
 * @throw cudf::logic_error if indices_column type is not INT32
 *
 * @param keys_column Column of unique, ordered values to use as the new dictionary column's keys.
 * @param indices_column Indices to use for the new dictionary column.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New dictionary column.
 */
std::unique_ptr<column> make_dictionary_column(
  column_view const& keys_column,
  column_view const& indices_column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Construct a dictionary column by taking ownership of the provided keys
 * and indices columns.
 *
 * @ingroup column_factories
 *
 * The keys_column and indices columns must contain no nulls.
 * It is assumed the elements in `keys_column` are unique and
 * are in a strict, total order. Meaning, `keys_column[i]` is ordered before
 * `keys_column[i+1]` for all `i in [0,n-1)` where `n` is the number of keys.
 *
 * The indices values must be in the range [0,keys_column.size()).
 *
 * @throw cudf::logic_error if keys_column or indices_column contains nulls
 * @throw cudf::logic_error if indices_column type is not INT32
 *
 * @param keys_column Column of unique, ordered values to use as the new dictionary column's keys.
 * @param indices_column Indices to use for the new dictionary column.
 * @param null_mask Null mask for the output column.
 * @param null_count Number of nulls for the output column.
 * @return New dictionary column.
 */
std::unique_ptr<column> make_dictionary_column(std::unique_ptr<column> keys_column,
                                               std::unique_ptr<column> indices_column,
                                               rmm::device_buffer&& null_mask,
                                               size_type null_count);

}  // namespace cudf
