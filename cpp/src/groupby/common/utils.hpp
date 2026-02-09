/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <memory>
#include <vector>

namespace cudf {
namespace groupby {
namespace detail {

template <typename RequestType>
inline std::vector<aggregation_result> extract_results(host_span<RequestType const> requests,
                                                       cudf::detail::result_cache& cache,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  std::vector<aggregation_result> results(requests.size());
  std::unordered_map<std::pair<column_view, std::reference_wrapper<aggregation const>>,
                     column_view,
                     cudf::detail::pair_column_aggregation_hash,
                     cudf::detail::pair_column_aggregation_equal_to>
    repeated_result;
  for (size_t i = 0; i < requests.size(); i++) {
    for (auto&& agg : requests[i].aggregations) {
      if (cache.has_result(requests[i].values, *agg)) {
        results[i].results.emplace_back(cache.release_result(requests[i].values, *agg));
        repeated_result[{requests[i].values, *agg}] = results[i].results.back()->view();
      } else {
        auto it = repeated_result.find({requests[i].values, *agg});
        if (it != repeated_result.end()) {
          results[i].results.emplace_back(std::make_unique<column>(it->second, stream, mr));
        } else {
          CUDF_FAIL("Cannot extract result from the cache");
        }
      }
    }
  }
  return results;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
