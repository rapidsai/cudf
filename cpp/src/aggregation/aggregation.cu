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

#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {
void initialize_with_identity(mutable_table_view& table,
                              host_span<cudf::aggregation::Kind const> aggs,
                              rmm::cuda_stream_view stream)
{
  // TODO: Initialize all the columns in a single kernel instead of invoking one
  // kernel per column
  for (size_type i = 0; i < table.num_columns(); ++i) {
    auto col = table.column(i);
    dispatch_type_and_aggregation(col.type(), aggs[i], identity_initializer{}, col, stream);
  }
}

}  // namespace detail
}  // namespace cudf
