/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
// The translation unit for reduction `max`

#include <cudf/detail/reduction_functions.hpp>
#include "simple.cuh"

std::unique_ptr<cudf::scalar> cudf::experimental::reduction::max(
    column_view const& col, cudf::data_type const output_dtype,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  using reducer = cudf::experimental::reduction::simple::element_type_dispatcher< cudf::experimental::reduction::op::max>;
  return cudf::experimental::type_dispatcher(col.type(), reducer(), col, output_dtype, mr, stream);
}
