/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

/**
 * @brief Main example function returns redacted strings column.
 *
 * This function returns a redacted version of the input `names` column
 * using the the `visibilities` column as in the following example
 * ```
 * names        visibility  --> redacted
 * John Doe     public          D John
 * Bobby Joe    private         X X
 * ```
 *
 * @param names First and last names separated with a single space
 * @param visibilities String values `public` or `private` only
 * @return Redacted strings column
 */
std::unique_ptr<cudf::column> redact_strings(cudf::column_view const& names,
                                             cudf::column_view const& visibilities);

/**
 * @brief Create CUDA memory resource
 */
auto make_cuda_mr() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

/**
 * @brief Create a pool device memory resource
 */
auto make_pool_mr()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda_mr());
}

/**
 * @brief Create memory resource for libcudf functions
 */
std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(std::string const& name)
{
  if (name == "pool") { return make_pool_mr(); }
  return make_cuda_mr();
}

/**
 * @brief Main for strings examples
 *
 * Command line parameters:
 * 1. CSV file name/path
 * 2. Memory resource (optional): 'pool' or 'cuda'
 *
 * The stdout includes the number of rows in the input and the output size in bytes.
 */
int main(int argc, char const** argv)
{
  if (argc < 2) {
    std::cout << "required parameter: csv-file-path\n";
    return 1;
  }

  auto const mr_name = std::string{argc > 2 ? std::string(argv[2]) : std::string("cuda")};
  auto resource      = create_memory_resource(mr_name);
  rmm::mr::set_current_device_resource(resource.get());

  auto const csv_file   = std::string{argv[1]};
  auto const csv_result = [csv_file] {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{csv_file}).header(-1);
    return cudf::io::read_csv(in_opts).tbl;
  }();
  auto const csv_table = csv_result->view();

  std::cout << "table: " << csv_table.num_rows() << " rows " << csv_table.num_columns()
            << " columns\n";

  auto st     = std::chrono::steady_clock::now();
  auto result = redact_strings(csv_table.column(0), csv_table.column(1));

  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - st;
  std::cout << "Wall time: " << elapsed.count() << " seconds\n";
  std::cout << "Output size " << result->view().child(1).size() << " bytes\n";

  return 0;
}
