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

#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/writers.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cudf {
namespace experimental {
namespace io {
namespace detail {
namespace csv {

using namespace cudf::io::csv;
using namespace cudf::io;

/**
 * @brief Implementation for CSV writer
 **/
class writer::impl {
  // CSV datasets are divided into fixed-size, independent stripes
  static constexpr uint32_t DEFAULT_STRIPE_SIZE = 64 * 1024 * 1024; // <- TODO: find right size!

  // CSV rows are divided into groups and assigned indexes for faster seeking
  static constexpr uint32_t DEFAULT_ROW_INDEX_STRIDE = 10000; // <- TODO: find the righ value

 public:
  /**
   * @brief Constructor with writer options.
   *
   * @param sink Output sink
   * @param options Settings for controlling behavior
   * @param mr Resource to use for device memory allocation
   **/
  explicit impl(std::unique_ptr<data_sink> sink, writer_options const& options,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Write an entire dataset to CSV format.
   *
   * @param table The set of columns
   * @param metadata The metadata associated with the table
   * @param stream Stream to use for memory allocation and kernels
   **/
  void write(table_view const& table, const table_metadata *metadata, cudaStream_t stream);
};

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
