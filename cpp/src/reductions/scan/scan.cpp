/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scan.hpp>
#include <cudf/reduction.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {

std::unique_ptr<column> scan(column_view const& input,
                             std::unique_ptr<aggregation> const& agg,
                             scan_type inclusive,
                             null_policy null_handling,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (agg->kind == aggregation::RANK) {
    CUDF_EXPECTS(inclusive == scan_type::INCLUSIVE,
                 "Unsupported rank aggregation operator for exclusive scan");
    return inclusive_rank_scan(input, rmm::cuda_stream_default, mr);
  }
  if (agg->kind == aggregation::DENSE_RANK) {
    CUDF_EXPECTS(inclusive == scan_type::INCLUSIVE,
                 "Unsupported dense rank aggregation operator for exclusive scan");
    return inclusive_dense_rank_scan(input, rmm::cuda_stream_default, mr);
  }

  return inclusive == scan_type::EXCLUSIVE
           ? detail::scan_exclusive(input, agg, null_handling, rmm::cuda_stream_default, mr)
           : detail::scan_inclusive(input, agg, null_handling, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
