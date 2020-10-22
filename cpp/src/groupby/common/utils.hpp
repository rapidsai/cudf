/*
 * Copyright (c) 2019-20, NVIDIA CORPORATION.
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

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/groupby.hpp>
#include <vector>

namespace cudf {
namespace groupby {
namespace detail {
inline std::vector<aggregation_result> extract_results(
  std::vector<aggregation_request> const& requests, cudf::detail::result_cache& cache)
{
  std::vector<aggregation_result> results(requests.size());

  for (size_t i = 0; i < requests.size(); i++) {
    for (auto&& agg : requests[i].aggregations) {
      results[i].results.emplace_back(cache.release_result(i, *agg));
    }
  }
  return results;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
