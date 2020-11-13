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
                                  rmm::mr::device_memory_resource* mr,
                                  cudaStream_t stream)
{
  auto const input_col = cudf::is_dictionary(col.type()) ? dictionary_column_view(col).keys() : col;
  CUDF_EXPECTS(input_col.type() == output_dtype, "min() operation requires matching output type");
  return cudf::type_dispatcher(input_col.type(),
                               simple::same_element_type_dispatcher<cudf::reduction::op::min>{},
                               input_col,
                               mr,
                               stream);
}

}  // namespace reduction
}  // namespace cudf
