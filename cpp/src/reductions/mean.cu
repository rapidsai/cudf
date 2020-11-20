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
// The translation unit for reduction `mean`

#include "compound.cuh"

#include <cudf/detail/reduction_functions.hpp>

#include <rmm/cuda_stream_view.hpp>

std::unique_ptr<cudf::scalar> cudf::reduction::mean(column_view const& col,
                                                    cudf::data_type const output_dtype,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  using reducer = cudf::reduction::compound::element_type_dispatcher<cudf::reduction::op::mean>;
  return cudf::type_dispatcher(
    col.type(), reducer(), col, output_dtype, /* ddof is not used for mean*/ 1, stream, mr);
}
