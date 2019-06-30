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
#include "aggregation_requests.hpp"

#include <cudf/cudf.h>
#include <cudf/binaryop.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table.hpp>
#include <utilities/column_utils.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>
#include "type_info.hpp"

#include <rmm/rmm.h>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace cudf {
namespace groupby {
namespace hash {

std::vector<AggRequestType> compound_to_simple(
    std::vector<AggRequestType> const& compound_requests) {
  // Contructs a mapping of every value column to the minimal set of simple
  // ops to be performed on that column
  std::unordered_map<gdf_column*, std::set<operators>> columns_to_ops;
  std::for_each(
      compound_requests.begin(), compound_requests.end(),
      [&columns_to_ops](std::pair<gdf_column const*, operators> pair) {
        gdf_column* col = const_cast<gdf_column*>(pair.first);
        CUDF_EXPECTS(col != nullptr, "Null column in aggregation request.");
        auto op = pair.second;
        // MEAN requires computing a COUNT and SUM aggregation and then doing
        // elementwise division
        if (op == MEAN) {
          columns_to_ops[col].insert(COUNT);
          columns_to_ops[col].insert(SUM);
        } else {
          columns_to_ops[col].insert(op);
        }
      });

  // Create minimal set of columns and simple operators
  std::vector<AggRequestType> simple_requests;
  for (auto& p : columns_to_ops) {
    auto col = p.first;
    std::set<operators>& ops = p.second;
    while (not ops.empty()) {
      simple_requests.emplace_back(col, *ops.begin());
      ops.erase(ops.begin());
    }
  }
  return simple_requests;
}

struct avg_result_type {
  template <typename SourceType>
  gdf_dtype operator()() {
    return cudf::gdf_dtype_of<target_type_t<SourceType, MEAN>>();
  }
};

gdf_column* compute_average(gdf_column sum, gdf_column count,
                            cudaStream_t stream) {
  CUDF_EXPECTS(sum.size == count.size,
               "Size mismatch between sum and count columns.");
  gdf_column* avg = new gdf_column{};

  avg->dtype = cudf::type_dispatcher(sum.dtype, avg_result_type{});
  avg->size = sum.size;
  RMM_TRY(RMM_ALLOC(&avg->data, sizeof(double) * sum.size, stream));
  if (cudf::is_nullable(sum) or cudf::is_nullable(count)) {
    RMM_TRY(RMM_ALLOC(
        &avg->valid,
        sizeof(gdf_size_type) * gdf_valid_allocation_size(sum.size), stream));
  }
  cudf::binary_operation(avg, &sum, &count, GDF_DIV);
  return avg;
}

table compute_original_requests(
    std::vector<AggRequestType> const& original_requests,
    std::vector<AggRequestType> const& simple_requests, table simple_outputs,
    cudaStream_t stream) {
  // Maps the requested simple aggregation to the resulting output column
  std::map<AggRequestType, gdf_column*> simple_requests_to_outputs;

  for (std::size_t i = 0; i < simple_requests.size(); ++i) {
    CUDF_EXPECTS(simple_outputs.get_column(i) != nullptr,
                 "Missing output column[" + std::to_string(i) + "]");

    simple_requests_to_outputs[simple_requests[i]] =
        simple_outputs.get_column(i);
  }

  std::vector<gdf_column*> final_value_columns;

  // Iterate requests. For any compound request, compute the compound result
  // from the corresponding simple requests
  for (auto const& req : original_requests) {
    if (req.second == MEAN) {
      auto found = simple_requests_to_outputs.find({req.first, SUM});
      CUDF_EXPECTS(found != simple_requests_to_outputs.end(),
                   "SUM request missing.");
      gdf_column* sum = found->second;

      CUDF_EXPECTS(sum != nullptr, "SUM column is null.");

      found = simple_requests_to_outputs.find({req.first, COUNT});
      CUDF_EXPECTS(found != simple_requests_to_outputs.end(),
                   "COUNT request missing.");
      gdf_column* count = found->second;

      CUDF_EXPECTS(count != nullptr, "COUNT column is null.");

      final_value_columns.push_back(compute_average(*sum, *count, stream));
    } else {
      // For non-compound requests, append the result to the final output
      // and remove it from the map
      auto found = simple_requests_to_outputs.find(req);
      CUDF_EXPECTS(found != simple_requests_to_outputs.end(),
                   "Aggregation missing!");

      CUDF_EXPECTS(found->second != nullptr,
                   "Simple aggregation result is null.");

      final_value_columns.push_back(found->second);
      simple_requests_to_outputs.erase(req);
    }
  }

  // Any remaining columns in the `simple_outputs` are intermediary columns used
  // to satisfy a compound request that should be deleted.
  for (auto& p : simple_requests_to_outputs) {
    gdf_column_free(p.second);
    delete p.second;
  }

  return cudf::table{final_value_columns};
}
}  // namespace hash
}  // namespace groupby
}  // namespace cudf