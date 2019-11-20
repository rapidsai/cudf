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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

namespace cudf {

namespace experimental {

std::unique_ptr<scalar>
quantile(column_view const& in,
         double quantile,
         interpolation interpolation,
         rmm::mr::device_memory_resource *mr,
         cudaStream_t stream)
{
    if (in.size() == 0) {
        return make_numeric_scalar(in.type(), stream, mr);
    }

    throw new std::runtime_error("not implemented");
}

std::vector<std::unique_ptr<scalar>>
quantiles(table_view const& in,
          double quantile,
          interpolation interpolation,
          rmm::mr::device_memory_resource *mr,
          cudaStream_t stream)
{
    std::vector<std::unique_ptr<scalar>> out(in.num_columns());

    std::transform(in.begin(), in.end(), out.begin(), [&](column_view const& in_column) {
        return experimental::quantile(in_column, quantile, interpolation, mr, stream);
    });

    return out;
}

 } // namespace experimental

 } // namespace cudf
