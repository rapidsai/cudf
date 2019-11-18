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

 #pragma once

 #include <cudf/table/table.hpp>
 #include <cudf/table/table_view.hpp>
 #include <cudf/quantiles.hpp>
 #include <cudf/scalar/scalar.hpp>
 #include <cudf/scalar/scalar_factories.hpp>
 
 namespace cudf {
     
 namespace experimental {
 
 /* @brief Computes the quantile of any sorted arithmetic column.
  *
  * @param[in] in                     Column from which quantile is computed.
  * @param[in] quantile_interpolation Strategy to obtain a quantile which falls
                                      between two points.
  *
  * @returns The quantile within range [0, 1]
  */
 std::unique_ptr<scalar> quantile(column_view const& in,
                                  double quantile,
                                  quantile_interpolation interpolation,
                                  cudaStream_t stream,
                                  rmm::mr::device_memory_resource *mr)
{
    if (in.size() == 0) {
        return make_numeric_scalar(in.type(), stream, mr);
    }
}
 
 } // namespace cudf
 
 } // namespace experimental
 