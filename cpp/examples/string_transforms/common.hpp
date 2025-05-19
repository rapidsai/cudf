/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief Main example function returns transformed string table.
 *
 * @param table Table to be transformed
 * @return Transformed result
 */
std::unique_ptr<cudf::column> transform(cudf::table_view const& table);

/**
 * @brief Create CUDA memory resource
 */
auto make_cuda_mr() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

/**
 * @brief Create a pool device memory resource
 */
auto make_pool_mr()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
    make_cuda_mr(), rmm::percent_of_free_device_memory(50));
}

/**
 * @brief Create memory resource for libcudf functions
 */
std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(std::string const& name)
{
  if (name == "pool") { return make_pool_mr(); }
  return make_cuda_mr();
}

void write_csv(cudf::table_view const& tbl_view, std::string const& file_path)
{
  auto sink_info = cudf::io::sink_info(file_path);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view).include_header(false);
  auto options   = builder.build();
  cudf::io::write_csv(options);
}

/**
 * @brief Main for strings examples
 *
 * Command line parameters:
 * 1. CSV file name/path
 * 2. Out file name/path
 * 3. Memory resource (optional): 'pool' or 'cuda'
 *
 * The stdout includes the number of rows in the input and the output size in bytes.
 */
int main(int argc, char const** argv)
{
  if (argc < 3) {
    std::cout << "required parameters: csv-file-path out-file-path\n";
    return 1;
  }

  auto const mr_name = std::string{argc >= 4 ? std::string(argv[3]) : std::string("cuda")};
  auto const out_csv = std::string{argv[2]};
  auto const in_csv  = std::string{argv[1]};
  auto resource      = create_memory_resource(mr_name);
  cudf::set_current_device_resource(resource.get());

  auto const csv_result = [in_csv] {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{in_csv}).header(0);
    return cudf::io::read_csv(in_opts).tbl;
  }();
  auto const csv_table = csv_result->view();

  std::cout << "table: " << csv_table.num_rows() << " rows " << csv_table.num_columns()
            << " columns\n";

  auto st     = std::chrono::steady_clock::now();
  auto result = transform(csv_table);

  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - st;
  std::cout << "Wall time: " << elapsed.count() << " seconds\n";

  std::vector<std::unique_ptr<cudf::column>> table_columns;
  table_columns.push_back(std::move(result));

  auto out_table = cudf::table(std::move(table_columns));

  write_csv(out_table, out_csv);

  return 0;
}
