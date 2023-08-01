/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/json.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

struct BlogExample : public cudf::test::BaseFixture {};

void write_json(cudf::table_view tbl, std::string path)
{
  // write the data for inspection
  auto sink_info = cudf::io::sink_info(path);
  auto builder2  = cudf::io::json_writer_options::builder(sink_info, tbl).lines(true);
  auto options2  = builder2.build();
  cudf::io::write_json(options2);
}

std::unique_ptr<cudf::table> nunique_func(cudf::table_view tbl)  // group nunique + filter > 1
{
  // do the nunique aggregation
  auto keys = cudf::table_view{{tbl.column(0)}};
  auto val  = tbl.column(1);
  cudf::groupby::groupby grpby_obj(keys);
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  auto agg = cudf::make_nunique_aggregation<cudf::groupby_aggregation>();
  requests[0].aggregations.push_back(std::move(agg));
  requests[0].values = val;
  auto agg_results   = grpby_obj.aggregate(requests);
  auto result_key    = std::move(agg_results.first);
  auto result_val    = std::move(agg_results.second[0].results[0]);
  std::vector<cudf::column_view> columns{result_key->get_column(0), *result_val};
  auto agg_v = cudf::table_view(columns);

  // filter out the keys with nunique > 1
  auto const op      = cudf::ast::ast_operator::EQUAL;
  auto literal_value = cudf::numeric_scalar<int32_t>(1);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref_1     = cudf::ast::column_reference(1);
  auto expression    = cudf::ast::operation(op, col_ref_1, literal);
  auto boolean_mask  = cudf::compute_column(agg_v, expression);
  auto filtered      = cudf::apply_boolean_mask(agg_v, boolean_mask->view());

  // semi join to gather only those keys from the original table
  auto join_indices      = cudf::left_semi_join(cudf::table_view{{tbl.column(0)}},
                                           cudf::table_view{{filtered->view().column(0)}});
  auto left_indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
  auto left_indices_col  = cudf::column_view{left_indices_span};
  auto filtered2         = cudf::gather(tbl, left_indices_col);

  write_json(*filtered2, "/home/nfs/dgala/cudf/cpp/tests/blog/unique_filter.json");

  return filtered2;
}

std::unique_ptr<cudf::table> max_func(cudf::table_view tbl)  // groupby max + filter >= 0.8
{
  // do the groupbymax aggregation
  auto keys = cudf::table_view{{tbl.column(0)}};
  auto val  = tbl.column(2);
  cudf::groupby::groupby grpby_obj(keys);
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  auto agg = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  requests[0].aggregations.push_back(std::move(agg));
  requests[0].values = val;
  auto agg_results   = grpby_obj.aggregate(requests);
  auto result_key    = std::move(agg_results.first);
  auto result_val    = std::move(agg_results.second[0].results[0]);
  std::vector<cudf::column_view> columns{result_key->get_column(0), *result_val};
  auto agg_v = cudf::table_view(columns);

  // filter out the keys with nunique > 1
  auto const op      = cudf::ast::ast_operator::GREATER_EQUAL;
  auto literal_value = cudf::numeric_scalar<double>(0.8);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref_1     = cudf::ast::column_reference(1);
  auto expression    = cudf::ast::operation(op, col_ref_1, literal);
  auto boolean_mask  = cudf::compute_column(agg_v, expression);
  auto filtered      = cudf::apply_boolean_mask(agg_v, boolean_mask->view());

  // semi join to gather only those keys from the original table
  auto join_indices      = cudf::left_semi_join(cudf::table_view{{tbl.column(0)}},
                                           cudf::table_view{{filtered->view().column(0)}});
  auto left_indices_span = cudf::device_span<cudf::size_type const>{*join_indices};
  auto left_indices_col  = cudf::column_view{left_indices_span};
  auto filtered2         = cudf::gather(tbl, left_indices_col);

  // write the data for inspection
  write_json(*filtered2, "/home/nfs/dgala/cudf/cpp/tests/blog/max_greater_filter.json");

  return filtered2;
}

void sort_func(cudf::table_view tbl)
{
  auto sorted_tbl = cudf::sort(tbl);

  write_json(*sorted_tbl, "/home/nfs/dgala/cudf/cpp/tests/blog/sort.json");
}

TEST_F(BlogExample, Test)
{
  // load the json from the example
  auto source_info = cudf::io::source_info("/home/nfs/dgala/cudf/cpp/tests/blog/example.json");
  auto builder     = cudf::io::json_reader_options::builder(source_info).lines(true);
  auto options     = builder.build();
  auto json        = cudf::io::read_json(options);
  auto tbl         = json.tbl->view();

  auto nunique_result = nunique_func(tbl);

  auto max_result = max_func(nunique_result->view());

  sort_func(max_result->view());
}
