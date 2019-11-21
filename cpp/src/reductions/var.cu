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

// The translation unit for reduction `variance`

#include "compound.cuh"
#include <cudf/detail/reduction_functions.hpp>

// @param[in] ddof Delta Degrees of Freedom used for `std`, `var`.
//                 The divisor used in calculations is N - ddof, where N
//                 represents the number of elements.

std::unique_ptr<cudf::scalar> cudf::experimental::reduction::variance(
    column_view const& col, cudf::data_type const output_dtype,
    cudf::size_type ddof, 
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  // TODO: add cuda version check when the fix is available
#if !defined(__CUDACC_DEBUG__)
  using reducer = cudf::experimental::reduction::compound::element_type_dispatcher< cudf::experimental::reduction::op::variance>;
  return cudf::experimental::type_dispatcher(col.type(), reducer(), col, output_dtype, ddof, mr, stream);
#else
  // workaround for bug 200529165 which causes compilation error only at device
  // debug build the bug will be fixed at cuda 10.2
  CUDF_FAIL("var/std reductions are not supported at debug build.");
#endif
}
