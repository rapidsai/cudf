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
using result_t  = std::unique_ptr<cudf::table>;

struct chunk_fn {
  cudf::io::text::data_chunk_source const& source;
  std::vector<result_t>& agg_data;
  rmm::cuda_stream_view stream;

  std::vector<cudf::io::text::byte_range_info> byte_ranges{};
  bool first_range{};

  void add_range(cudf::io::text::byte_range_info br)
  {
    byte_ranges.push_back(br);
    if (!first_range) { first_range = (br.offset() == 0); }
  }

  void operator()()
  {
    using namespace std::chrono_literals;
    // std::cout << std::this_thread::get_id() << "=" << first_range << std::endl;
    if (!first_range) {
      std::this_thread::sleep_for(350ms);  // add some fixed delay
    }

    // process each byte range assigned to this thread
    for (auto& br : byte_ranges) {
      // load byte-range from the file into 2 strings columns (cities, temps)
      auto splits = [&] {
        cudf::io::text::parse_options options{br, false};
        auto raw_data_column = cudf::io::text::multibyte_split(source, "\n", options, stream);
        auto const sv        = cudf::strings_column_view(raw_data_column->view());
        auto const delimiter = cudf::string_scalar{";", true, stream};
        return cudf::strings::split(sv, delimiter, 1, stream);
      }();

      // convert temps strings to floats
      auto temps  = cudf::strings::to_floats(cudf::strings_column_view(splits->view().column(1)),
                                            cudf::data_type{cudf::type_id::FLOAT32},
                                            stream);
      auto cities = std::move(splits->release().front());

      // compute aggregations on this chunk
      std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
      aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
      aggregations.emplace_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
      aggregations.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
      aggregations.emplace_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
      auto result = compute_results(cities->view(), temps->view(), std::move(aggregations), stream);

      // store sorted results
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

  auto const input_file   = std::string{argv[1]};
  auto const divider      = (argc < 3) ? 10 : std::stoi(std::string(argv[2]));
  auto const thread_count = (argc < 4) ? 2 : std::stoi(std::string(argv[3]));

  std::cout << "input:   " << input_file << std::endl;
  std::cout << "chunks:  " << divider << std::endl;
  std::cout << "threads: " << thread_count << std::endl;

  auto const mr_name = std::string("pool");  // "cuda"
  auto resource      = create_memory_resource(mr_name);
  rmm::mr::set_current_device_resource(resource.get());
  auto stream = cudf::get_default_stream();

  std::filesystem::path p = input_file;
  auto const file_size    = std::filesystem::file_size(p);

  auto start = std::chrono::steady_clock::now();

  auto byte_ranges  = cudf::io::text::create_byte_range_infos_consecutive(file_size, divider);
  auto const source = cudf::io::text::make_source_from_file(input_file);

  // use multiple threads assigning a stream per thread
  auto stream_pool = rmm::cuda_stream_pool(thread_count);
  std::vector<std::vector<result_t>> chunk_results(thread_count);

  std::vector<chunk_fn> chunks;
  for (auto& cr : chunk_results) {
    chunks.emplace_back(chunk_fn{*source, cr, stream_pool.get_stream()});
  }
  for (std::size_t i = 0; i < byte_ranges.size(); ++i) {
    chunks[i % thread_count].add_range(byte_ranges[i]);
  }
  std::vector<std::thread> threads;
  for (auto& c : chunks) {
    threads.emplace_back(std::thread{c});
  }
  for (auto& t : threads) {
    t.join();
  }

  // in case some APIs are still running on the default stream
  stream.synchronize();

  // combine each thread's agg data into a single vector
  std::vector<result_t> agg_data(divider);
  auto begin = agg_data.begin();
  for (auto& c : chunk_results) {
    std::transform(c.begin(), c.end(), begin, [](auto&& d) { return std::move(d); });
    begin += c.size();
  }

  // now aggregate the threads' aggregate results
  auto results = compute_final_aggregates(agg_data, stream);
  std::cout << "number of keys: " << results->num_rows() << std::endl;

  auto elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "process time: " << (elapsed.count() / 1e9) << " seconds\n";

  return 0;
}
