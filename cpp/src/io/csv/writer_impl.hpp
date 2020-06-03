/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

/**
 * @file writer_impl.hpp
 * @brief cuDF-IO CSV writer class implementation header
 */

#pragma once

#include "csv.h"
#include "csv_gpu.h"

#include <cudf/strings/strings_column_view.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/writers.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cudf {
namespace io {
namespace detail {
namespace csv {

using namespace cudf::io::csv;
using namespace cudf::io;

/**
 * @brief Implementation for CSV writer
 **/
class writer::impl {
 public:
  /**
   * @brief Constructor with writer options.
   *
   * @param sink Output sink
   * @param options Settings for controlling behavior
   * @param mr Device memory resource to use for device memory allocation
   **/
  impl(std::unique_ptr<data_sink> sink,
       writer_options const& options,
       rmm::mr::device_memory_resource* mr);

  /**
   * @brief Write an entire dataset to CSV format.
   *
   * @param table The set of columns
   * @param metadata The metadata associated with the table
   * @param stream CUDA stream used for device memory operations and kernel launches.
   **/
  void write(table_view const& table,
             const table_metadata* metadata = nullptr,
             cudaStream_t stream            = nullptr);

  /**
   * @brief Write the header of a CSV format.
   *
   * @param table The set of columns
   * @param metadata The metadata associated with the table
   * @param stream CUDA stream used for device memory operations and kernel launches.
   **/
  void write_chunked_begin(table_view const& table,
                           const table_metadata* metadata = nullptr,
                           cudaStream_t stream            = nullptr);

  /**
   * @brief Write dataset to CSV format without header.
   *
   * @param strings_column Subset of columns converted to string to be written.
   * @param metadata The metadata associated with the table
   * @param stream CUDA stream used for device memory operations and kernel launches.
   **/
  void write_chunked(strings_column_view const& strings_column,
                     const table_metadata* metadata = nullptr,
                     cudaStream_t stream            = nullptr);

  /**
   * @brief Write footer of CSV format (typically, empty).
   *
   * @param table The set of columns
   * @param metadata The metadata associated with the table
   * @param stream CUDA stream used for device memory operations and kernel launches.
   **/
  void write_chunked_end(table_view const& table,
                         const table_metadata* metadata = nullptr,
                         cudaStream_t stream            = nullptr)
  {
    // purposely no-op (for now);
  }

 private:
  std::unique_ptr<data_sink> out_sink_;
  rmm::mr::device_memory_resource* mr_ = nullptr;
  writer_options const options_;
};

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace cudf
