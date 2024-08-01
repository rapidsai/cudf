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

#include <cudf/io/csv.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

cudf::io::table_with_metadata read_csv(std::string const& file_path)
{
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::csv_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_csv(options);
}

void write_csv(cudf::table_view const& tbl_view, std::string const& file_path)
{
  auto sink_info = cudf::io::sink_info(file_path);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view);
  auto options   = builder.build();
  cudf::io::write_csv(options);
}

// process the transactions table with all keep options
void process_keep_options(cudf::table_view transactions_info_table)
{
  // keep any transacation if there are duplicates
  auto keep_any_table =
    cudf::distinct(transactions_info_table, {0}, cudf::duplicate_keep_option::KEEP_ANY);
  write_csv(*keep_any_table, "transactions_keep_any.csv");

  // keep the first transacation if there are multiple transactions by a single customer
  auto keep_first_table =
    cudf::distinct(transactions_info_table, {0}, cudf::duplicate_keep_option::KEEP_FIRST);
  write_csv(*keep_first_table, "transactions_keep_first.csv");

  // keep the last transacation if there are multiple transactions by a single customer
  auto keep_last_table =
    cudf::distinct(transactions_info_table, {0}, cudf::duplicate_keep_option::KEEP_LAST);
  write_csv(*keep_last_table, "transactions_keep_last.csv");

  // keep the customers with only single transaction
  auto keep_none_table =
    cudf::distinct(transactions_info_table, {0}, cudf::duplicate_keep_option::KEEP_NONE);
  write_csv(*keep_none_table, "transactions_keep_none.csv");
}

int main(int argc, char** argv)
{
  // Construct a CUDA memory resource using RAPIDS Memory Manager (RMM)
  // This is the default memory resource for libcudf for allocating device memory.
  rmm::mr::cuda_memory_resource cuda_mr{};
  // Construct a memory pool using the CUDA memory resource
  // Using a memory pool for device memory allocations is important for good performance in libcudf.
  // The pool defaults to allocating half of the available GPU memory.
  rmm::mr::pool_memory_resource mr{&cuda_mr, rmm::percent_of_free_device_memory(50)};

  // Set the pool resource to be used by default for all device memory allocations
  // Note: It is the user's responsibility to ensure the `mr` object stays alive for the duration of
  // it being set as the default
  // Also, call this before the first libcudf API call to ensure all data is allocated by the same
  // memory resource.
  rmm::mr::set_current_device_resource(&mr);

  // Read data
  auto transactions_table_with_metadata = read_csv("transactions.csv");

  // process the transactions table with all keep options
  process_keep_options(*transactions_table_with_metadata.tbl);

  return 0;
}
