/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf/utilities/traits.hpp>

namespace cudf::reduction::detail {

std::unique_ptr<scalar> argmin(column_view const& input,
                               data_type const output_dtype,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  auto const dispatch_type =
    is_dictionary(input.type()) ? dictionary_column_view(input).indices().type() : input.type();
  return type_dispatcher(dispatch_type,
                         simple::detail::arg_minmax_dispatcher<aggregation::ARGMIN>{},
                         input,
                         output_dtype,
                         stream,
                         mr);
}

}  // namespace cudf::reduction::detail
