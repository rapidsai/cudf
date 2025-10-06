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

#include "parquet_io.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>

#include <algorithm>
#include <iostream>
#include <thread>

namespace cudf {
namespace examples {

void write_parquet_file(const std::string& filepath,
                        cudf::table_view table_view,
                        rmm::cuda_stream_view stream)
{
  auto sink_info = cudf::io::sink_info(filepath);
  auto builder   = cudf::io::parquet_writer_options::builder(sink_info, table_view);

  // Create metadata for better compression
  auto table_metadata = cudf::io::table_input_metadata{table_view};
  for (cudf::size_type i = 0; i < table_view.num_columns(); ++i) {
    table_metadata.column_metadata[i].set_name("column_" + std::to_string(i));
  }

  auto options = builder.metadata(table_metadata).compression(cudf::io::compression_type::SNAPPY);
  cudf::io::write_parquet(options.build(), stream);
  stream.synchronize();
}

void write_task::operator()()
{
  std::string filepath = output_dir + "/data_" + std::to_string(file_id) + ".parquet";
  std::cout << "Writing file: " << filepath << std::endl;
  write_parquet_file(filepath, table_views[file_id], stream);
  std::cout << "Completed writing: " << filepath << std::endl;
}

void read_task::operator()()
{
  std::vector<std::unique_ptr<cudf::table>> tables_this_thread;

  // Process files assigned to this thread
  for (size_t file_idx = thread_id; file_idx < filepaths.size(); file_idx += thread_count) {
    std::cout << "Thread " << thread_id << " reading: " << filepaths[file_idx] << std::endl;

    auto source_info = cudf::io::source_info(filepaths[file_idx]);
    auto builder     = cudf::io::parquet_reader_options::builder(source_info);
    auto options     = builder.build();

    tables_this_thread.push_back(cudf::io::read_parquet(options, stream).tbl);
  }

  // Concatenate tables read by this thread
  if (tables_this_thread.size() == 1) {
    tables[thread_id] = std::move(tables_this_thread[0]);
  } else if (tables_this_thread.size() > 1) {
    std::vector<cudf::table_view> table_views;
    for (auto const& tbl : tables_this_thread) {
      table_views.push_back(tbl->view());
    }
    tables[thread_id] = cudf::concatenate(table_views, stream);
  }

  stream.synchronize();
}

std::vector<std::unique_ptr<cudf::table>> read_parquet_files_multithreaded(const std::vector<std::string>& filepaths,
                                                                           int thread_count,
                                                                           rmm::cuda_stream_pool& stream_pool)
{
  std::vector<std::unique_ptr<cudf::table>> tables(thread_count);
  std::vector<read_task> read_tasks;
  std::vector<std::thread> threads;

  // Create read tasks
  for (int tid = 0; tid < thread_count; ++tid) {
    read_tasks.emplace_back(
      read_task{filepaths, tables, tid, thread_count, stream_pool.get_stream()});
  }

  // Launch threads
  for (auto& task : read_tasks) {
    threads.emplace_back(task);
  }

  // Wait for completion
  for (auto& thread : threads) {
    thread.join();
  }

  // Remove empty tables
  tables.erase(std::remove_if(tables.begin(),
                             tables.end(),
                             [](const std::unique_ptr<cudf::table>& tbl) { return tbl == nullptr; }),
              tables.end());

  return tables;
}

}  // namespace examples
}  // namespace cudf
