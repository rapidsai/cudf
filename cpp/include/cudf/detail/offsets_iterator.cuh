/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/normalizing_iterator.cuh>

namespace cudf {
namespace detail {

/**
 * @brief The index normalizing input iterator.
 *
 * This is an iterator that can be used for index types (integers) without
 * requiring a type-specific instance. It can be used for any iterator
 * interface for reading an array of integer values of type
 * int8, int16, int32, int64, uint8, uint16, uint32, or uint64.
 * Reading specific elements always return a `size_type` integer.
 *
 * Use the indexalator_factory to create an appropriate input iterator
 * from a column_view.
 *
 * Example input iterator usage.
 * @code
 *  auto begin = indexalator_factory::create_input_iterator(gather_map);
 *  auto end   = begin + gather_map.size();
 *  auto result = detail::gather( source, begin, end, IGNORE, stream, mr );
 * @endcode
 *
 * @code
 *  auto begin = indexalator_factory::create_input_iterator(indices);
 *  auto end   = begin + indices.size();
 *  auto result = thrust::find(thrust::device, begin, end, size_type{12} );
 * @endcode
 */
using input_offsetsalator = input_normalator<int64_t>;

/**
 * @brief The index normalizing output iterator.
 *
 * This is an iterator that can be used for index types (integers) without
 * requiring a type-specific instance. It can be used for any iterator
 * interface for writing an array of integer values of type
 * int8, int16, int32, int64, uint8, uint16, uint32, or uint64.
 * Setting specific elements always accept `size_type` integer values.
 *
 * Use the indexalator_factory to create an appropriate output iterator
 * from a mutable_column_view.
 *
 * Example output iterator usage.
 * @code
 *  auto result_itr = indexalator_factory::create_output_iterator(indices->mutable_view());
 *  thrust::lower_bound(rmm::exec_policy(stream),
 *                      input->begin<Element>(),
 *                      input->end<Element>(),
 *                      values->begin<Element>(),
 *                      values->end<Element>(),
 *                      result_itr,
 *                      thrust::less<Element>());
 * @endcode
 */
using output_offsetsalator = output_normalator<int64_t>;

}  // namespace detail
}  // namespace cudf
