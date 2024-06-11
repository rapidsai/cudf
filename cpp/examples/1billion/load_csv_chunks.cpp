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

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>
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

std::unique_ptr<cudf::table> load_chunk1(std::string const& input_file,
                                         cudf::size_type start_row,
                                         cudf::size_type size)
{
  auto start = std::chrono::steady_clock::now();
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{input_file})
      .header(-1)
      .delimiter(';')
      .doublequote(false)
      .skiprows(start_row)
      .nrows(size)
      .dtypes(std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::STRING},
                                           cudf::data_type{cudf::type_id::FLOAT32}})
      .na_filter(false);
  auto result = cudf::io::read_csv(in_opts).tbl;

  elapsed_t elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "CSV(" << start_row << "," << result->num_rows() << "): " << elapsed.count()
            << " seconds\n";

  return result;
}

std::unique_ptr<cudf::table> load_chunk(std::string const& input_file,
                                        std::size_t start,
                                        std::size_t size)
{
  auto st = std::chrono::steady_clock::now();
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
  auto result = cudf::io::read_csv(in_opts).tbl;

  elapsed_t elapsed = std::chrono::steady_clock::now() - st;
  std::cout << "CSV(" << start << "," << result->num_rows() << "): " << elapsed.count()
            << " seconds\n";

  return result;
}

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

  std::filesystem::path p = csv_file;
  auto file_size          = std::filesystem::file_size(p);
  std::cout << "file size = " << file_size << std::endl;

  std::vector<std::unique_ptr<cudf::table>> results;

  std::size_t chunk_size     = file_size / 25;
  std::size_t start_row      = 0;
  cudf::size_type total_rows = 0;
  do {
    auto const input_table = load_chunk(csv_file, start_row, chunk_size);
    auto const read_rows   = input_table->num_rows();
    std::cout << "input rows: " << read_rows << std::endl;
    if (read_rows == 0) break;

    auto const cities = input_table->view().column(0);
    auto const temps  = input_table->view().column(1);
    auto scv          = cudf::strings_column_view(cities);
    std::cout << "Cities column: " << scv.chars_size(stream) << " bytes\n";

    auto start_tm     = std::chrono::steady_clock::now();
    auto result       = compute_results(cities, temps);
    elapsed_t elapsed = std::chrono::steady_clock::now() - start_tm;
    std::cout << "Process time: " << elapsed.count() << " seconds\n";
    std::cout << result->num_rows() << " rows" << std::endl;

    result = cudf::sort_by_key(result->view(), result->view().select({0}));
    results.emplace_back(std::move(result));
    // start_row += read_rows;
    start_row += chunk_size;
    if (start_row + chunk_size > file_size) { chunk_size = file_size - start_row; }
    total_rows += read_rows;
  } while (start_row < file_size && chunk_size > 0);

  std::cout << "total input " << total_rows << " rows" << std::endl;

  // aggregate the aggregate results
  std::cout << "results count: " << results.size() << std::endl;

  std::vector<cudf::column_view> mins, maxes, sums, counts;
  for (auto& tbl : results) {
    auto const tv = tbl->view();
    mins.push_back(tv.column(1));
    maxes.push_back(tv.column(2));
    sums.push_back(tv.column(4));
    counts.push_back(tv.column(5));
  }

  auto all_mins   = cudf::interleave_columns(cudf::table_view{mins});
  auto all_maxes  = cudf::interleave_columns(cudf::table_view{maxes});
  auto all_sums   = cudf::interleave_columns(cudf::table_view{sums});
  auto all_counts = cudf::interleave_columns(cudf::table_view{counts});

  auto num_keys     = results.front()->num_rows();
  auto offsets      = cudf::sequence(static_cast<cudf::size_type>(num_keys) + 1,
                                cudf::numeric_scalar<cudf::size_type>(0),
                                cudf::numeric_scalar<cudf::size_type>(results.size()));
  auto offsets_span = cudf::device_span<cudf::size_type const>(offsets->view());

  auto min_agg   = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
  auto each_mins = cudf::segmented_reduce(
    all_mins->view(), offsets_span, *min_agg, all_mins->type(), cudf::null_policy::EXCLUDE);

  auto max_agg    = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
  auto each_maxes = cudf::segmented_reduce(
    all_maxes->view(), offsets_span, *max_agg, all_maxes->type(), cudf::null_policy::EXCLUDE);

  auto sum_agg   = cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>();
  auto each_sums = cudf::segmented_reduce(
    all_sums->view(), offsets_span, *sum_agg, all_sums->type(), cudf::null_policy::EXCLUDE);
  auto each_counts = cudf::segmented_reduce(
    all_counts->view(), offsets_span, *sum_agg, all_counts->type(), cudf::null_policy::EXCLUDE);

  auto each_means = cudf::binary_operation(
    each_sums->view(), each_counts->view(), cudf::binary_operator::DIV, each_sums->type());
  // cudf::test::print(each_means->view());
  std::cout << each_means->size() << std::endl;
  return 0;
}
