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

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <queue>

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental {

/**
 * @addtogroup io_readers
 * @{
 * @file
 */

/**
 * @brief The chunked parquet reader class to read a Parquet source iteratively in a series of
 * tables, chunk by chunk. Each chunk is prepended with a row index column built using the specified
 * row group offsets and row counts. The resultant table chunk is filtered using the supplied
 * serialized roaring64 bitmap deletion vector and returned
 *
 * This class is designed to address the reading issue when reading very large Parquet source such
 * that the row count exceeds the cudf column size limit or if there are device memory constraints.
 * By reading the source content by chunks using this class, each chunk is guaranteed to have its
 * sizes stay within the given limit. Note that the given memory limits do not account for the
 * device memory needed to deserialize and construct the roaring64 bitmap deletion vector that stays
 * alive throughout the the lifetime of the reader.
 */
class chunked_parquet_reader {
 public:
  //! Forward declaration of the opaque wrapper of cuco's 64-bit roaring bitmap
  struct roaring_bitmap_impl;

  /**
   * @brief Constructor for the chunked reader
   *
   * Requires the same arguments as the `cudf::io::parquet::experimental::read_parquet()`, and an
   * additional parameter to specify the size byte limit of the output table chunk produced.
   *
   * @param chunk_read_limit Byte limit on the returned table chunk size, `0` if there is no limit
   * @param options Parquet reader options
   * @param serialized_roaring64 Host span of `portable` serialized 64-bit roaring bitmap
   * @param row_group_offsets Host span of row offsets of each row group
   * @param row_group_num_rows Host span of number of rows in each row group
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  chunked_parquet_reader(
    std::size_t chunk_read_limit,
    parquet_reader_options const& options,
    cudf::host_span<cuda::std::byte const> serialized_roaring64,
    cudf::host_span<size_t const> row_group_offsets,
    cudf::host_span<size_type const> row_group_num_rows,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Constructor for the chunked reader
   *
   * Requires the same arguments as `cudf::io::parquet::experimental::read_parquet()`, with
   * additional parameters to specify the size byte limit of the output table chunk produced, and a
   * byte limit on the amount of temporary memory to use when reading. The `pass_read_limit` affects
   * how many row groups we can read at a time by limiting the amount of memory dedicated to
   * decompression space. The `pass_read_limit` is a hint, not an absolute limit - if a single row
   * group cannot fit within the limit given, it will still be loaded. Also note that the
   * `pass_read_limit` does not include the memory to deserialize and construct the roaring64 bitmap
   * deletion vector that stays alive throughout the the lifetime of the reader.
   *
   * @param chunk_read_limit Byte limit on the returned table chunk size, `0` if there is no limit
   * @param pass_read_limit Byte limit on the amount of memory used for decompressing and decoding
   * data, `0` if there is no limit
   * @param options Parquet reader options
   * @param serialized_roaring64 Host span of `portable` serialized 64-bit roaring bitmap
   * @param row_group_offsets Host span of row offsets of each row group
   * @param row_group_num_rows Host span of number of rows in each row group
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  chunked_parquet_reader(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    parquet_reader_options const& options,
    cudf::host_span<cuda::std::byte const> serialized_roaring64,
    cudf::host_span<size_t const> row_group_offsets,
    cudf::host_span<size_type const> row_group_num_rows,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Destructor, destroying the internal reader instance and the roaring bitmap deletion
   * vector
   */
  ~chunked_parquet_reader();

  /**
   * @brief Check if there is any data in the given source that has not yet been read
   *
   * @return Boolean value indicating if there is any data left to be read
   */
  [[nodiscard]] bool has_next() const;

  /**
   * @brief Read a chunk of table from the Parquet source, prepend an index column to it, and
   * filters the resultant table chunk using the 64-bit roaring bitmap deletion vector, if provided
   *
   * The sequence of returned tables, if concatenated by their order, guarantees to form a complete
   * dataset as reading the entire given source at once.
   *
   * An empty table will be returned if the given source is empty, or all the data in the source has
   * been read and returned by the previous calls.
   *
   * @return An output `cudf::table` along with its metadata
   */
  [[nodiscard]] table_with_metadata read_chunk();

 private:
  std::unique_ptr<cudf::io::chunked_parquet_reader> _reader;
  std::queue<size_t> _row_group_row_offsets;
  std::queue<size_type> _row_group_row_counts;
  std::unique_ptr<roaring_bitmap_impl> _deletion_vector;
  size_t _start_row;
  bool _is_unspecified_row_group_data;
  rmm::cuda_stream_view _stream;
  rmm::device_async_resource_ref _mr;
  rmm::device_async_resource_ref _table_mr;
};

/**
 * @brief Reads a table from parquet source, prepends an index column to it, deserializes the
 * roaring64 deletion vector and applies it to the read table
 *
 * Reads a table from a parquet source, builds a row index column to the table using the specified
 * row group offsets and row counts and prepends it to the table, deserializes the specified
 * roaring64 deletion vector and applies it to the read table. If the row group offsets and row
 * counts are empty, the index column is simply a sequence of UINT64 from 0 to the total number of
 * rows in the table. If the serialized roaring64 bitmap span is empty, the read table (prepended
 * with the index column) is returned as is.
 *
 * @ingroup io_readers
 *
 * @param options Parquet reader options
 * @param serialized_roaring64 Host span of `portable` serialized 64-bit roaring bitmap
 * @param row_group_offsets Host span of row index offsets for each row group
 * @param row_group_num_rows Host span of number of rows in each row group
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned table
 *
 * @return Read table with a prepended index column filtered using the deletion vector, along with
 * its metadata
 */
table_with_metadata read_parquet(
  parquet_reader_options const& options,
  cudf::host_span<cuda::std::byte const> serialized_roaring64,
  cudf::host_span<size_t const> row_group_offsets,
  cudf::host_span<size_type const> row_group_num_rows,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
