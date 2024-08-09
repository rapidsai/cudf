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
#include <cudf/io/datasource.hpp>
#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/split/split.hpp>
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

  auto const input_file = std::string{argv[1]};
  std::cout << "input:   " << input_file << std::endl;

  auto const mr_name = std::string("pool");  // "cuda"
  auto resource      = create_memory_resource(mr_name);
  rmm::mr::set_current_device_resource(resource.get());
  auto stream = cudf::get_default_stream();

  auto start = std::chrono::steady_clock::now();

  auto const source = cudf::io::text::make_source_from_file(input_file);
  cudf::io::text::parse_options options;
  options.strip_delimiters = false;
  auto raw_data_column     = cudf::io::text::multibyte_split(*source, "\n", options);

  elapsed_t elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "file load time: " << elapsed.count() << " seconds" << std::endl;
  std::cout << "input rows: " << raw_data_column->size() << std::endl;

  auto sv        = cudf::strings_column_view(raw_data_column->view());
  auto delimiter = cudf::string_scalar{";"};
  auto splits    = cudf::strings::split(sv, delimiter, 1);

  raw_data_column->release();  // no longer needed

  auto temps  = cudf::strings::to_floats(cudf::strings_column_view(splits->view().column(1)),
                                        cudf::data_type{cudf::type_id::FLOAT32});
  auto cities = std::move(splits->release().front());

  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  aggregations.emplace_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  aggregations.emplace_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());

  auto result = compute_results(cities->view(), temps->view(), std::move(aggregations));

  elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "number of keys: " << result->num_rows() << std::endl;
  std::cout << "process time: " << elapsed.count() << " seconds\n";

  return 0;
}
