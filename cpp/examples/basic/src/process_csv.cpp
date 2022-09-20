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

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

cudf::io::table_with_metadata read_csv(std::string const& file_path)
{
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::csv_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_csv(options);
}

void write_csv(cudf::table_view const& tbl_view, std::string const& file_path)
{
  auto sink_info = cudf::io::sink_info(file_path);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view);
  auto options   = builder.build();
  cudf::io::write_csv(options);
}

std::vector<cudf::groupby::aggregation_request> make_single_aggregation_request(
  std::unique_ptr<cudf::groupby_aggregation>&& agg, cudf::column_view value)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].aggregations.push_back(std::move(agg));
  requests[0].values = value;
  return requests;
}

std::unique_ptr<cudf::table> average_closing_price(cudf::table_view stock_info_table)
{
  // Schema: | Company | Date | Open | High | Low | Close | Volume |
  auto keys = cudf::table_view{{stock_info_table.column(0)}};  // Company
  auto val  = stock_info_table.column(5);                      // Close

  // Compute the average of each company's closing price with entire column
  cudf::groupby::groupby grpby_obj(keys);
  auto requests =
    make_single_aggregation_request(cudf::make_mean_aggregation<cudf::groupby_aggregation>(), val);

  auto agg_results = grpby_obj.aggregate(requests);

  // Assemble the result
  auto result_key = std::move(agg_results.first);
  auto result_val = std::move(agg_results.second[0].results[0]);
  std::vector<cudf::column_view> columns{result_key->get_column(0), *result_val};
  return std::make_unique<cudf::table>(cudf::table_view(columns));
}

int main(int argc, char** argv)
{
  // Read data
  auto stock_table_with_metadata = read_csv("4stock_5day.csv");

  // Process
  auto result = average_closing_price(*stock_table_with_metadata.tbl);

  // Write out result
  write_csv(*result, "4stock_5day_avg_close.csv");

  return 0;
}
