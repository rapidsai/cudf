/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Return a fixed-width value from column.
 *
 * Retrieves the specified value from device memory. This function
 * synchronizes the stream.
 *
 * @throw cudf::logic_error if `col_view` is not a fixed-width column
 * @throw cudf::logic_error if `element_index < 0 or >= col_view.size()`
 *
 * @tparam T Fixed-width type to return.
 * @param col_view The column to retrieve the element from.
 * @param element_index The specific element to retrieve
 * @param stream The stream to use for copying the value to the host.
 * @return Value from the `col_view[element_index]`
 */
template <typename T>
T get_value(column_view const& col_view, size_type element_index, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(cudf::is_fixed_width(col_view.type()), "get_value supports only fixed-width types");
  CUDF_EXPECTS(data_type(type_to_id<T>()) == col_view.type(), "get_value data type mismatch");
  CUDF_EXPECTS(element_index >= 0 && element_index < col_view.size(),
               "invalid element_index value");
  return cudf::detail::make_host_vector_sync(
           device_span<T const>{col_view.data<T>() + element_index, 1}, stream)
    .front();
}

}  // namespace detail
}  // namespace cudf
