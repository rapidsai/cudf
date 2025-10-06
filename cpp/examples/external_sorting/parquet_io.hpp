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

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <vector>

/**
 * @file parquet_io.hpp
 * @brief Parquet I/O utilities for external sorting example
 *
 * This header provides utilities for reading and writing parquet files
 * with multithreaded support for better performance on large datasets.
 */

namespace cudf {
namespace examples {

/**
 * @brief Write a table to a parquet file
 *
 * Writes the given table to a parquet file with Snappy compression.
 * Column names are automatically generated as "column_0", "column_1", etc.
 *
 * @param filepath Path to the output parquet file
 * @param table_view View of the table to write
 * @param stream CUDA stream for operations
 */
void write_parquet_file(const std::string& filepath,
                        cudf::table_view table_view,
                        rmm::cuda_stream_view stream);

/**
 * @brief Functor for multithreaded parquet writing
 *
 * This functor can be used with std::thread to write parquet files
 * in parallel. Each thread writes one file using its assigned stream.
 */
struct write_task {
  std::string const& output_dir;
  std::vector<cudf::table_view> const& table_views;
  int const file_id;
  rmm::cuda_stream_view stream;

  void operator()();
};

/**
 * @brief Functor for multithreaded parquet reading
 *
 * This functor can be used with std::thread to read parquet files
 * in parallel. Each thread processes a subset of files and concatenates
 * them if multiple files are assigned to the same thread.
 */
struct read_task {
  std::vector<std::string> const& filepaths;
  std::vector<std::unique_ptr<cudf::table>>& tables;
  int const thread_id;
  int const thread_count;
  rmm::cuda_stream_view stream;

  void operator()();
};

/**
 * @brief Read multiple parquet files using multithreading
 *
 * Reads a list of parquet files using multiple threads for improved
 * performance. Files are distributed across threads in round-robin fashion.
 * Each thread concatenates its assigned files into a single table.
 *
 * @param filepaths List of parquet file paths to read
 * @param thread_count Number of threads to use for reading
 * @param stream_pool CUDA stream pool for thread synchronization
 * @return Vector of tables, one per thread (empty tables are filtered out)
 */
std::vector<std::unique_ptr<cudf::table>> read_parquet_files_multithreaded(const std::vector<std::string>& filepaths,
                                                                           int thread_count,
                                                                           rmm::cuda_stream_pool& stream_pool);

}  // namespace examples
}  // namespace cudf
