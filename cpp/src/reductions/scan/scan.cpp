/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scan.hpp>
#include <cudf/reduction.hpp>

namespace cudf {
namespace detail {
namespace {
std::unique_ptr<column> scan(column_view const& input,
                             scan_aggregation const& agg,
                             scan_type inclusive,
                             null_policy null_handling,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  if (agg.kind == aggregation::RANK) {
    CUDF_EXPECTS(inclusive == scan_type::INCLUSIVE,
                 "Rank aggregation operator requires an inclusive scan");
    auto const& rank_agg = static_cast<cudf::detail::rank_aggregation const&>(agg);
    if (rank_agg._method == rank_method::MIN) {
      if (rank_agg._percentage == rank_percentage::NONE) {
        return inclusive_rank_scan(input, stream, mr);
      } else if (rank_agg._percentage == rank_percentage::ONE_NORMALIZED) {
        return inclusive_one_normalized_percent_rank_scan(input, stream, mr);
      }
    } else if (rank_agg._method == rank_method::DENSE) {
      return inclusive_dense_rank_scan(input, stream, mr);
    }
    CUDF_FAIL("Unsupported rank aggregation method for inclusive scan");
  }

  return inclusive == scan_type::EXCLUSIVE
           ? detail::scan_exclusive(input, agg, null_handling, stream, mr)
           : detail::scan_inclusive(input, agg, null_handling, stream, mr);
}

}  // namespace
}  // namespace detail

std::unique_ptr<column> scan(column_view const& input,
                             scan_aggregation const& agg,
                             scan_type inclusive,
                             null_policy null_handling,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::scan(input, agg, inclusive, null_handling, stream, mr);
}

}  // namespace cudf
