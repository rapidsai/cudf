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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

using elapsed_t = std::chrono::duration<double>;

int main(int argc, char const** argv)
{
  if (argc < 2) {
    std::cout << "required parameter: csv-file-path\n";
    return 1;
  }

  auto const csv_file = std::string{argv[1]};
  auto const mr_name  = std::string{argc > 2 ? std::string(argv[2]) : std::string("cuda")};
  auto resource       = create_memory_resource(mr_name);
  rmm::mr::set_current_device_resource(resource.get());
  auto stream = cudf::get_default_stream();

  auto start            = std::chrono::steady_clock::now();
  auto const csv_result = [csv_file] {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{csv_file})
        .header(-1)
        .delimiter(';')
        .doublequote(false)
        .dtypes(std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::STRING},
                                             cudf::data_type{cudf::type_id::FLOAT32}})
        .na_filter(false);
    return cudf::io::read_csv(in_opts).tbl;
  }();
  elapsed_t elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "CSV load time: " << elapsed.count() << " seconds\n";
  auto const csv_table = csv_result->view();
  std::cout << "input rows: " << csv_table.num_rows() << std::endl;

  auto const cities = csv_table.column(0);
  auto const temps  = csv_table.column(1);
  auto scv          = cudf::strings_column_view(cities);
  std::cout << "Cities column: " << scv.chars_size(stream) << " bytes\n";

  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  aggregations.emplace_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  aggregations.emplace_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());

  start       = std::chrono::steady_clock::now();
  auto result = compute_results(cities, temps, std::move(aggregations));
  elapsed     = std::chrono::steady_clock::now() - start;
  std::cout << "Process time: " << elapsed.count() << " seconds\n";

  result = cudf::sort_by_key(result->view(), result->view().select({0}));
  std::cout << result->num_rows() << " rows" << std::endl;

  return 0;
}
