/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "simple.cuh"

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace reduction {
namespace detail {

std::unique_ptr<cudf::scalar> product(column_view const& col,
                                      cudf::data_type const output_dtype,
                                      std::optional<std::reference_wrapper<scalar const>> init,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(
    cudf::is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type(),
    simple::detail::element_type_dispatcher<op::product>{},
    col,
    output_dtype,
    init,
    stream,
    mr);
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf
