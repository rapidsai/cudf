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
#include "common.hpp"
#include "groupby_results.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

using elapsed_t = std::chrono::duration<double>;

std::unique_ptr<cudf::table> load_chunk(std::string const& input_file,
                                        std::size_t start,
                                        std::size_t size)
{
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{input_file})
      .header(-1)
      .delimiter(';')
      .doublequote(false)
      .byte_range_offset(start)
      .byte_range_size(size)
      .dtypes(std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::STRING},
                                           cudf::data_type{cudf::type_id::FLOAT32}})
      .na_filter(false);
  return cudf::io::read_csv(in_opts).tbl;
}

int main(int argc, char const** argv)
{
  if (argc < 2) {
    std::cout << "required parameter: csv-file-path\n";
    return 1;
  }

  auto const input_file = std::string{argv[1]};
  auto const divider    = (argc < 3) ? 25 : std::stoi(std::string(argv[2]));

  std::cout << "input:   " << input_file << std::endl;
  std::cout << "chunks:  " << divider << std::endl;

  auto const mr_name = std::string("pool");  // "cuda"
  auto resource      = create_memory_resource(mr_name);
  rmm::mr::set_current_device_resource(resource.get());
  auto stream = cudf::get_default_stream();

  std::filesystem::path p = input_file;
  auto const file_size    = std::filesystem::file_size(p);

  auto start = std::chrono::steady_clock::now();

  std::vector<std::unique_ptr<cudf::table>> agg_data;
  std::size_t chunk_size     = file_size / divider + ((file_size % divider) != 0);
  std::size_t start_pos      = 0;
  cudf::size_type total_rows = 0;
  do {
    auto const input_table = load_chunk(input_file, start_pos, chunk_size);
    auto const read_rows   = input_table->num_rows();
    if (read_rows == 0) break;

    auto const cities = input_table->view().column(0);
    auto const temps  = input_table->view().column(1);

    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
    aggregations.emplace_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
    aggregations.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    aggregations.emplace_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
    auto result = compute_results(cities, temps, std::move(aggregations));

    agg_data.emplace_back(cudf::sort_by_key(result->view(), result->view().select({0})));
    start_pos += chunk_size;
    if (start_pos + chunk_size > file_size) { chunk_size = file_size - start_pos; }
    total_rows += read_rows;
  } while (start_pos < file_size && chunk_size > 0);

  // now aggregate the aggregate results
  auto results = compute_final_aggregates(agg_data);

  elapsed_t elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "number of keys = " << results->num_rows() << std::endl;
  std::cout << "process time: " << elapsed.count() << " seconds\n";

  return 0;
}
