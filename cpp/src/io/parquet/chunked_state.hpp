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
 * @brief definition for chunked state structure used by Parquet writer
 */

#pragma once

#include <cudf/io/detail/parquet.hpp>
#include <io/parquet/parquet.hpp>

namespace cudf {
namespace io {

enum class SingleWriteMode : bool { YES, NO };

/**
 * @brief Chunked writer state struct. Contains various pieces of information
 *        needed that span the begin() / write() / end() call process.
 */
struct pq_chunked_state {
  /// The writer to be used
  std::unique_ptr<cudf::io::detail::parquet::writer> wp;
  /// Cuda stream to be used
  cudaStream_t stream;
  /// Overall file metadata.  Filled in during the process and written during write_chunked_end()
  cudf::io::parquet::FileMetaData md;
  /// current write position for rowgroups/chunks
  std::size_t current_chunk_offset;
  /// optional user metadata
  table_metadata_with_nullability user_metadata_with_nullability;
  /// special parameter only used by detail::write() to indicate that we are guaranteeing
  /// a single table write.  this enables some internal optimizations.
  table_metadata const* user_metadata = nullptr;
  /// only used in the write_chunked() case. copied from the (optionally) user supplied
  /// argument to write_parquet_chunked_begin()
  bool single_write_mode;

  pq_chunked_state() = default;

  pq_chunked_state(table_metadata const* metadata,
                   SingleWriteMode mode = SingleWriteMode::NO,
                   cudaStream_t str     = 0)
    : user_metadata{metadata}, single_write_mode{mode == SingleWriteMode::YES}, stream{str}
  {
  }
};

}  // namespace io
}  // namespace cudf
