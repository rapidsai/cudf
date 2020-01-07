/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

#include <cudf/unary.hpp>
#include <cudf/column/column_factories.hpp>

namespace cudf {
namespace experimental {
namespace detail{

/**
 * @brief Creates a column of `BOOL8` elements by applying a predicate to every element between [`begin, `end`)
 * `true` indicates the value is satisfies the predicate and `false` indicates it doesn't.
 *
 * @tparam InputIterator Iterator type for `begin` and `end`
 * @tparam Predicate A predicator type which will be evaludated
 * @param begin Begining of the sequence of elements
 * @param end End of the sequence of elements
 * @param p Predicate to be applied to each element in `[begin,end)`
 * @param mr Optional, The resource to use for all allocations
 * @param stream Optional CUDA stream on which to execute kernels
 *
 * @returns std::unique_ptr<cudf::column> A column of type `BOOL8,` with `true` representing predicate is satisfied.
 */

template <typename InputIterator, typename Predicate>
std::unique_ptr<column> true_if(InputIterator begin, InputIterator end,
                           size_type size, Predicate p,
                           rmm::mr::device_memory_resource * mr =
                               rmm::mr::get_default_resource(),
                           cudaStream_t stream = 0) {
    auto output = make_numeric_column(data_type(BOOL8), size, UNALLOCATED, stream, mr);
    auto output_mutable_view = output->mutable_view();
    auto output_data = output_mutable_view.data<cudf::experimental::bool8>();

    thrust::transform(rmm::exec_policy(stream)->on(stream), begin, end, output_data, p);

    return output;
}

/**
 * @brief Performs unary op on all values in column
 *
 * @param input A `column_view` as input
 * @param op operation to perform
 * @param mr Optional, The resource to use for all allocations
 * @param stream Optional CUDA stream on which to execute kernels
 *
 * @returns std::unique_ptr<cudf::column> Result of the operation
 */
std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::experimental::unary_op op,
                                              rmm::mr::device_memory_resource* mr =
                                              rmm::mr::get_default_resource(),
                                              cudaStream_t stream = 0);


/**
 * @brief  Casts data from dtype specified in input to dtype specified in output.
 * Supports only fixed-width types.
 *
 * @param column_view Input column
 * @param out_type Desired datatype of output column
 * @param mr Optional, The resource to use for all allocations
 * @param stream Optional CUDA stream on which to execute kernels
 *
 * @returns unique_ptr<column> Result of the cast operation
 * @throw cudf::logic_error if `out_type` is not a fixed-width type
 */
std::unique_ptr<column> cast(column_view const& input,
                             data_type type,
                             rmm::mr::device_memory_resource* mr =
                                rmm::mr::get_default_resource(),
                             cudaStream_t stream = 0);

} // namespace detail
} // namespace experimental
} // namespace cudf
