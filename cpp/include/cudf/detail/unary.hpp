/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @brief Creates a column of `type_id::BOOL8` elements by applying a predicate to every element
 * between
 * [`begin, `end`) `true` indicates the value is satisfies the predicate and `false` indicates it
 * doesn't.
 *
 * @tparam InputIterator Iterator type for `begin` and `end`
 * @tparam Predicate A predicator type which will be evaluated
 * @param begin Beginning of the sequence of elements
 * @param end End of the sequence of elements
 * @param p Predicate to be applied to each element in `[begin,end)`
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column of type `type_id::BOOL8,` with `true` representing predicate is satisfied.
 */

template <typename InputIterator, typename Predicate>
std::unique_ptr<column> true_if(InputIterator begin,
                                InputIterator end,
                                size_type size,
                                Predicate p,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  auto output =
    make_numeric_column(data_type(type_id::BOOL8), size, mask_state::UNALLOCATED, stream, mr);
  auto output_mutable_view = output->mutable_view();
  auto output_data         = output_mutable_view.data<bool>();

  thrust::transform(rmm::exec_policy_nosync(stream), begin, end, output_data, p);

  return output;
}

/**
 * @copydoc cudf::unary_operation
 */
std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::unary_operator op,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::is_valid
 */
std::unique_ptr<cudf::column> is_valid(cudf::column_view const& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::cast
 */
std::unique_ptr<column> cast(column_view const& input,
                             data_type type,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::is_nan
 */
std::unique_ptr<column> is_nan(cudf::column_view const& input,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::is_not_nan
 */
std::unique_ptr<column> is_not_nan(cudf::column_view const& input,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
