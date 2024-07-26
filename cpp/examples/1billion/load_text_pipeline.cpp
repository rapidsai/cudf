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

#include <rmm/cuda_stream_pool.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using elapsed_t = std::chrono::duration<double>;

struct chunk_fn {
  cudf::io::text::data_chunk_source const& source;
  std::vector<std::unique_ptr<cudf::table>>& agg_data;
  rmm::cuda_stream_view& stream;
  std::vector<cudf::io::text::byte_range_info> byte_ranges{};

  template <typename Itr>
  void add_ranges(Itr begin, Itr end)
  {
    byte_ranges.insert(byte_ranges.end(), begin, end);
  }

  void add_range(cudf::io::text::byte_range_info const& br)
  {
    byte_ranges.insert(byte_ranges.end(), br);
  }

  void operator()()
  {
    for (auto& br : byte_ranges) {
      auto splits = [&] {
        cudf::io::text::parse_options options{br, false};
        auto raw_data_column = cudf::io::text::multibyte_split(source, "\n", options, stream);
        auto const sv        = cudf::strings_column_view(raw_data_column->view());
        auto const delimiter = cudf::string_scalar{";", true, stream};
        return cudf::strings::split(sv, delimiter, 1, stream);
      }();

      auto temps  = cudf::strings::to_floats(cudf::strings_column_view(splits->view().column(1)),
                                            cudf::data_type{cudf::type_id::FLOAT32},
                                            stream);
      auto cities = std::move(splits->release().front());

      std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
      aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
      aggregations.emplace_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
      aggregations.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      aggregations.emplace_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());

      auto result = compute_results(cities->view(), temps->view(), std::move(aggregations), stream);
      agg_data.emplace_back(
        cudf::sort_by_key(result->view(), result->view().select({0}), {}, {}, stream));
    }
    // done with this stream
    stream.synchronize_no_throw();
  }
};

int main(int argc, char const** argv)
{
  if (argc < 2) {
    std::cout << "required parameter: csv-file-path\n";
    return 1;
  }

  auto const input_file = std::string{argv[1]};
  auto const mr_name    = std::string{argc > 2 ? std::string(argv[2]) : std::string("cuda")};
  auto resource         = create_memory_resource(mr_name);
  rmm::mr::set_current_device_resource(resource.get());
  auto stream = cudf::get_default_stream();

  std::filesystem::path p = input_file;
  auto const file_size    = std::filesystem::file_size(p);

  int constexpr divider = 10;
  auto byte_ranges      = cudf::io::text::create_byte_range_infos_consecutive(file_size, divider);
  auto const source     = cudf::io::text::make_source_from_file(input_file);

  // use 2 threads
  auto stream_pool = rmm::cuda_stream_pool(2);
  auto stream1     = stream_pool.get_stream();
  std::vector<std::unique_ptr<cudf::table>> agg_data1;
  chunk_fn chunk1{*source, agg_data1, stream1};
  auto stream2 = stream_pool.get_stream();
  std::vector<std::unique_ptr<cudf::table>> agg_data2;
  chunk_fn chunk2{*source, agg_data2, stream2};

  // chunk1.add_ranges(byte_ranges.begin(), byte_ranges.begin() + (divider / 10));
  // chunk2.add_ranges(byte_ranges.begin() + (divider / 10), byte_ranges.end());
  for (std::size_t i = 0; i < byte_ranges.size(); ++i) {
    if (i % 1) {
      chunk1.add_range(byte_ranges[i]);
    } else {
      chunk2.add_range(byte_ranges[i]);
    }
  }

  std::thread t1(chunk1);
  std::thread t2(chunk2);
  t1.join();
  t2.join();

  // some APIs still run on the default stream
  stream.synchronize();

  std::vector<std::unique_ptr<cudf::table>> agg_data(agg_data1.size() + agg_data2.size());
  auto begin = agg_data.begin();
  std::transform(agg_data1.begin(), agg_data1.end(), begin, [](auto&& d) { return std::move(d); });
  begin += agg_data1.size();
  std::transform(agg_data2.begin(), agg_data2.end(), begin, [](auto&& d) { return std::move(d); });

  // now aggregate the aggregate results
  auto results = compute_final_aggregates(agg_data, stream);
  std::cout << "number of keys = " << results->num_rows() << std::endl;
  std::cout << "chunks = " << divider << std::endl;

  return 0;
}
