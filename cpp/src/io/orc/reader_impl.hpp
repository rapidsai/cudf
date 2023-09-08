/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include "aggregate_orc_metadata.hpp"
#include "orc.hpp"
#include "orc_gpu.hpp"

#include <io/utilities/column_buffer.hpp>
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/orc.hpp>
#include <cudf/io/orc.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cudf::io::detail::orc {
using namespace cudf::io::orc;

namespace {
struct reader_column_meta;
}

/**
 * @brief Implementation for ORC reader.
 */
class reader::impl {
 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   *
   * @param sources Dataset sources
   * @param options Settings for controlling reading behavior
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::vector<std::unique_ptr<datasource>>&& sources,
                orc_reader_options const& options,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Read an entire set or a subset of data and returns a set of columns
   *
   * @param skip_rows Number of rows to skip from the start
   * @param num_rows_opt Optional number of rows to read
   * @param stripes Indices of individual stripes to load if non-empty
   * @return The set of columns along with metadata
   */
  table_with_metadata read(uint64_t skip_rows,
                           std::optional<size_type> const& num_rows_opt,
                           std::vector<std::vector<size_type>> const& stripes);

 private:
  rmm::cuda_stream_view const _stream;
  rmm::mr::device_memory_resource* const _mr;

  std::vector<std::unique_ptr<datasource>> const _sources;  // Unused but owns data for `_metadata`
  cudf::io::orc::detail::aggregate_orc_metadata _metadata;
  cudf::io::orc::detail::column_hierarchy const _selected_columns;  // Need to be after _metadata

  data_type const _timestamp_type;  // Override output timestamp resolution
  bool const _use_index;            // Enable or disable attempt to use row index for parsing
  bool const _use_np_dtypes;        // Enable or disable the conversion to numpy-compatible dtypes
  std::vector<std::string> const _decimal128_columns;   // Control decimals conversion
  std::unique_ptr<reader_column_meta> const _col_meta;  // Track of orc mapping and child details
};

}  // namespace cudf::io::detail::orc
