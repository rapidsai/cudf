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

#include <cudf_test/debug_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <chrono>
#include <filesystem>
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
  auto const divider    = (argc < 3) ? 10 : std::stoi(std::string(argv[2]));

  std::cout << "input:   " << input_file << std::endl;
  std::cout << "chunks:  " << divider << std::endl;

  auto const mr_name = std::string("pool");  // "cuda"
  auto resource      = create_memory_resource(mr_name);
  rmm::mr::set_current_device_resource(resource.get());
  auto stream = cudf::get_default_stream();

  std::filesystem::path p = input_file;
  auto const file_size    = std::filesystem::file_size(p);

  auto byte_ranges  = cudf::io::text::create_byte_range_infos_consecutive(file_size, divider);
  auto const source = cudf::io::text::make_source_from_file(input_file);

  std::vector<std::unique_ptr<cudf::table>> agg_data;
  for (auto& br : byte_ranges) {
    auto splits = [&] {
      cudf::io::text::parse_options options{br, false};
      auto raw_data_column = cudf::io::text::multibyte_split(*source, "\n", options);
      auto const sv        = cudf::strings_column_view(raw_data_column->view());
      auto const delimiter = cudf::string_scalar{";"};
      return cudf::strings::split(sv, delimiter, 1);
    }();

    auto temps  = cudf::strings::to_floats(cudf::strings_column_view(splits->view().column(1)),
                                          cudf::data_type{cudf::type_id::FLOAT32});
    auto cities = std::move(splits->release().front());

    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
    aggregations.emplace_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
    aggregations.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    aggregations.emplace_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());

    auto result = compute_results(cities->view(), temps->view(), std::move(aggregations));
    agg_data.emplace_back(cudf::sort_by_key(result->view(), result->view().select({0})));
  }

  // now aggregate the aggregate results
  auto results = compute_final_aggregates(agg_data);
  std::cout << "number of keys = " << results->num_rows() << std::endl;

  return 0;
}
