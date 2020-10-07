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
 * @file chunked_state.hpp
 * @brief definition for chunked state structure used by ORC writer
 */

#pragma once

#include "orc.h"

#include <cudf/io/data_sink.hpp>
#include <cudf/io/detail/orc.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cudf {
namespace io {
/**
 * @brief Chunked writer state struct. Contains various pieces of information
 *        needed that span the begin() / write() / end() call process.
 */
struct orc_chunked_state {
  /// The writer to be used
  std::unique_ptr<detail::orc::writer> wp;
  /// Cuda stream to be used
  cudaStream_t stream;
  /// Overall file metadata.  Filled in during the process and written during write_chunked_end()
  cudf::io::orc::FileFooter ff;
  cudf::io::orc::Metadata md;
  /// current write position for rowgroups/chunks
  size_t current_chunk_offset;
  /// optional user metadata
  table_metadata const* user_metadata = nullptr;
  /// only used in the write_chunked() case. copied from the (optionally) user supplied
  /// argument to write_chunked_begin()
  table_metadata_with_nullability user_metadata_with_nullability;
  /// special parameter only used by detail::write() to indicate that we are guaranteeing
  /// a single table write.  this enables some internal optimizations.
  bool single_write_mode = false;
};

}  // namespace io
}  // namespace cudf
