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
#include <cudf/copying.hpp>
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
    std::cout
      << "insufficient argurments.\n\t\tin-csv-path out-csv-path [num_rows [memory_resource]]\n";
    return 1;
  }

  auto const in_csv   = std::string{argv[1]};
  auto const out_csv  = std::string{argv[2]};
  auto const num_rows = argc > 3 ? std::optional{std::stoi(std::string(argv[3]))} : std::nullopt;
  auto const memory_resource_name =
    std::string{argc > 5 ? std::string(argv[5]) : std::string("cuda")};

  auto resource = create_memory_resource(memory_resource_name);
  cudf::set_current_device_resource(resource.get());

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{in_csv}).header(0);
  auto input = cudf::io::read_csv(in_opts).tbl;

  if (num_rows.has_value() && input->get_column(0).size() != num_rows.value()) {
    input = cudf::sample(*input, num_rows.value(), cudf::sample_with_replacement::TRUE);
  }

  auto table_view = input->view();

  // TODO(lamarrr): make the transform return a table instead since the data is now sampled

  auto start  = std::chrono::steady_clock::now();
  auto result = transform(table_view);

  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

  std::vector<std::unique_ptr<cudf::column>> output_columns;
  output_columns.emplace_back(std::move(result));

  auto output = cudf::table(std::move(output_columns));

  write_csv(output, out_csv);

  std::cout << "Wall time: " << elapsed.count() << " seconds\n"
            << "Table: " << table_view.num_rows() << " rows " << table_view.num_columns()
            << " columns" << std::endl;

  return 0;
}
