/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cudf/utilities/default_stream.hpp>

namespace cudf {

std::unique_ptr<column> scan(column_view const& input,
                             std::unique_ptr<scan_aggregation> const& agg,
                             scan_type inclusive,
                             null_policy null_handling,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (agg->kind == aggregation::RANK) {
    CUDF_EXPECTS(inclusive == scan_type::INCLUSIVE,
                 "Rank aggregation operator requires an inclusive scan");
    auto const& rank_agg = dynamic_cast<cudf::detail::rank_aggregation const&>(*agg);
    if (rank_agg._method == rank_method::MIN) {
      if (rank_agg._percentage == rank_percentage::NONE) {
        return inclusive_rank_scan(input, cudf::default_stream_value, mr);
      } else if (rank_agg._percentage == rank_percentage::ONE_NORMALIZED) {
        return inclusive_one_normalized_percent_rank_scan(input, cudf::default_stream_value, mr);
      }
    } else if (rank_agg._method == rank_method::DENSE) {
      return inclusive_dense_rank_scan(input, cudf::default_stream_value, mr);
    }
    CUDF_FAIL("Unsupported rank aggregation method for inclusive scan");
  }

  return inclusive == scan_type::EXCLUSIVE
           ? detail::scan_exclusive(input, agg, null_handling, cudf::default_stream_value, mr)
           : detail::scan_inclusive(input, agg, null_handling, cudf::default_stream_value, mr);
}

}  // namespace cudf
