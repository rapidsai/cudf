/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/detail/reduction_functions.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <reductions/simple.cuh>

namespace cudf {
namespace reduction {

std::unique_ptr<cudf::scalar> min(column_view const& col,
                                  data_type const output_dtype,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  auto const input_type =
    cudf::is_dictionary(col.type()) ? cudf::dictionary_column_view(col).keys().type() : col.type();
  CUDF_EXPECTS(input_type == output_dtype, "min() operation requires matching output type");
  auto const dispatch_type = cudf::is_dictionary(col.type())
                               ? cudf::dictionary_column_view(col).indices().type()
                               : col.type();
  return cudf::type_dispatcher(dispatch_type,
                               simple::same_element_type_dispatcher<cudf::reduction::op::min>{},
                               col,
                               stream,
                               mr);
}

}  // namespace reduction
}  // namespace cudf
