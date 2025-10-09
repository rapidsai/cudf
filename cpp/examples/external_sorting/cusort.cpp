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

#include "cusort.hpp"

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>

namespace cudf {
namespace examples {

std::unique_ptr<cudf::column> sample_splitters(cudf::table_view const& table_view,
                                               cudf::size_type num_splitters,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (table_view.num_rows() == 0 || num_splitters <= 0) {
    // Return empty column of the same type as the first column
    return cudf::empty_like(table_view.column(0));
  }
  
  // Sort this table by first column
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER};
  
  auto sorted_indices = cudf::sorted_order(table_view.select({0}), column_order, null_precedence, stream, mr);
  // TODO: sample
  return sorted_indices;
}

}  // namespace examples
}  // namespace cudf
