/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "flatten_single_pass_aggs.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <memory>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace cudf::groupby::detail::hash {

class groupby_simple_aggregations_collector final
  : public cudf::detail::simple_aggregations_collector {
 public:
  using cudf::detail::simple_aggregations_collector::visit;

  std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                  cudf::detail::min_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(col_type.id() == type_id::STRING ? make_argmin_aggregation()
                                                    : make_min_aggregation());
    return aggs;
  }

  std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                  cudf::detail::max_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(col_type.id() == type_id::STRING ? make_argmax_aggregation()
                                                    : make_max_aggregation());
    return aggs;
  }

  std::vector<std::unique_ptr<aggregation>> visit(data_type col_type,
                                                  cudf::detail::mean_aggregation const&) override
  {
    (void)col_type;
    CUDF_EXPECTS(is_fixed_width(col_type), "MEAN aggregation expects fixed width type");
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(make_sum_aggregation());
    // COUNT_VALID
    aggs.push_back(make_count_aggregation());

    return aggs;
  }

  std::vector<std::unique_ptr<aggregation>> visit(data_type,
                                                  cudf::detail::var_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(make_sum_aggregation());
    // COUNT_VALID
    aggs.push_back(make_count_aggregation());

    return aggs;
  }

  std::vector<std::unique_ptr<aggregation>> visit(data_type,
                                                  cudf::detail::std_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(make_sum_aggregation());
    // COUNT_VALID
    aggs.push_back(make_count_aggregation());

    return aggs;
  }

  std::vector<std::unique_ptr<aggregation>> visit(
    data_type, cudf::detail::correlation_aggregation const&) override
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(make_sum_aggregation());
    // COUNT_VALID
    aggs.push_back(make_count_aggregation());

    return aggs;
  }
};

// flatten aggs to filter in single pass aggs
std::tuple<table_view,
           cudf::detail::host_vector<aggregation::Kind>,
           std::vector<std::unique_ptr<aggregation>>>
flatten_single_pass_aggs(host_span<aggregation_request const> requests,
                         rmm::cuda_stream_view stream)
{
  std::vector<column_view> columns;
  std::vector<std::unique_ptr<aggregation>> aggs;
  auto agg_kinds = cudf::detail::make_empty_host_vector<aggregation::Kind>(requests.size(), stream);

  for (auto const& request : requests) {
    auto const& agg_v = request.aggregations;

    std::unordered_set<aggregation::Kind> agg_kinds_set;
    auto insert_agg = [&](column_view const& request_values, std::unique_ptr<aggregation>&& agg) {
      if (agg_kinds_set.insert(agg->kind).second) {
        agg_kinds.push_back(agg->kind);
        aggs.push_back(std::move(agg));
        columns.push_back(request_values);
      }
    };

    auto values_type = cudf::is_dictionary(request.values.type())
                         ? cudf::dictionary_column_view(request.values).keys().type()
                         : request.values.type();
    for (auto&& agg : agg_v) {
      groupby_simple_aggregations_collector collector;

      for (auto& agg_s : agg->get_simple_aggregations(values_type, collector)) {
        insert_agg(request.values, std::move(agg_s));
      }
    }
  }

  return std::make_tuple(table_view(columns), std::move(agg_kinds), std::move(aggs));
}

}  // namespace cudf::groupby::detail::hash
