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

#include <cudf/groupby.hpp>
#include <cudf/io/json.hpp>
#include <cudf/join.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

#include <chrono>
#include <iostream>
#include <string>

std::unique_ptr<cudf::table> read_json(std::string filepath)
{
  auto source_info = cudf::io::source_info(filepath);
  auto builder     = cudf::io::json_reader_options::builder(source_info).lines(true);
  auto options     = builder.build();
  auto json        = cudf::io::read_json(options);
  return std::move(json.tbl);
}

void write_json(cudf::table_view tbl, std::string filepath)
{
  // write the data for inspection
  auto sink_info = cudf::io::sink_info(filepath);
  auto builder2  = cudf::io::json_writer_options::builder(sink_info, tbl).lines(true);
  auto options2  = builder2.build();
  cudf::io::write_json(options2);
}

std::unique_ptr<cudf::table> deduplication_hash(cudf::column_view col)
{
  auto tbl = cudf::table_view{{col}};

  // Get count for each key
  auto keys = cudf::table_view{{tbl.column(0)}};
  auto val  = tbl.column(0);
  cudf::groupby::groupby grpby_obj(keys);
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  requests[0].aggregations.push_back(std::move(agg));
  requests[0].values = val;
  auto agg_results   = grpby_obj.aggregate(requests);
  auto result_key    = std::move(agg_results.first);
  auto result_val    = std::move(agg_results.second[0].results[0]);
  std::vector<cudf::column_view> columns{result_key->get_column(0), *result_val};
  auto agg_v = cudf::table_view(columns);

  // Join on keys to get
  return std::make_unique<cudf::table>(agg_v);
}

std::unique_ptr<cudf::table> deduplication_sort(cudf::column_view col)
{
  auto tbl = cudf::table_view{{col}};

  // Get count for each key
  auto keys = cudf::table_view{{tbl.column(0)}};
  auto val  = tbl.column(0);
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

  // Join on keys to get
  return std::make_unique<cudf::table>(agg_v);
}

/**
 * @brief Main for nested_types examples
 *
 * Command line parameters:
 * 1. JSON input file name/path (default: "example.json")
 * 2. `hash` for hash based deduplication or `sort` for sort based deduplication (default: "hash")
 * 3. JSON output file name/path (default: "hash_output.json")
 *
 * The stdout includes the number of rows in the input and the output size in bytes.
 */
int main(int argc, char const** argv)
{
  std::string input_filepath;
  std::string algorithm;
  std::string output_filepath;
  if (argc < 2) {
    input_filepath  = "example.json";
    algorithm       = "hash";
    output_filepath = "hash_output.json";
  } else if (argc == 4) {
    input_filepath  = argv[1];
    algorithm       = argv[2];
    output_filepath = argv[3];
  } else {
    std::cout << "Either provide all command-line arguments, or none to use defaults" << std::endl;
    return 1;
  }

  // read input file
  auto tbl = read_json(input_filepath);

  auto st = std::chrono::steady_clock::now();

  // alg here
  std::unique_ptr<cudf::table> result;
  if (algorithm == "hash") {
    result = deduplication_hash(tbl->view().column(0));
  } else {
    result = deduplication_sort(tbl->view().column(0));
  }

  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - st;
  std::cout << "Wall time: " << elapsed.count() << " seconds\n";

  write_json(result->view(), output_filepath);

  return 0;
}
