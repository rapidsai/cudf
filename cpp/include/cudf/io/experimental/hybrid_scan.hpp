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

/**
 * @file parquet.hpp
 */

#pragma once

#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace experimental::io {

namespace parquet {

namespace detail {

/**
 * @brief Internal experimental Parquet reader optimized for hybrid scan
 */
class impl;

}  // namespace detail

/**
 * @brief The experimental parquet reader class to optimize reading parquet files subject to
 *        highly selective filters (Hybrid Scan operation).
 *
 * This class is designed to address the gap between the available and exploited optimization
 * techniques to speed up reading Parquet files subject to highly selective filters (Hybrid scan
 * operation). This class reads file contents in two passes, where the first pass optimizes reading
 * only the filter columns and the second pass optimizes reading the predicate columns.
 */
class hybrid_scan_reader {
 public:
  hybrid_scan_reader();

  hybrid_scan_reader(cudf::host_span<uint8_t const> footer_bytes,
                     cudf::host_span<uint8_t const> page_index_bytes,
                     cudf::io::parquet_reader_options const& options);

  ~hybrid_scan_reader();

  [[nodiscard]] std::vector<size_type> get_valid_row_groups(
    cudf::io::parquet_reader_options const& options) const;

  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_stats(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  [[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                          std::vector<cudf::io::text::byte_range_info>>
  get_secondary_filters(cudf::host_span<std::vector<size_type> const> row_group_indices,
                        cudf::io::parquet_reader_options const& options) const;

  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_dictionary_pages(
    std::vector<rmm::device_buffer>& dictionary_page_data,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  [[nodiscard]] std::vector<std::vector<size_type>> filter_row_groups_with_bloom_filters(
    std::vector<rmm::device_buffer>& bloom_filter_data,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  [[nodiscard]] std::vector<std::vector<bool>> filter_data_pages_with_stats(
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream) const;

  [[nodiscard]] cudf::io::table_with_metadata materialize_filter_columns(
    cudf::mutable_column_view input_rows,
    cudf::host_span<std::vector<size_type> const> row_group_indices,
    std::vector<rmm::device_buffer> column_chunk_bytes,
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream);

 private:
  std::unique_ptr<detail::impl> _impl;
};

}  // namespace parquet

// To be defined
struct hybrid_scan_metadata;

// API # 1
[[nodiscard]] std::unique_ptr<parquet::hybrid_scan_reader> make_hybrid_scan_reader(
  cudf::host_span<uint8_t const> footer_bytes,
  cudf::host_span<uint8_t const> page_index_bytes,
  cudf::io::parquet_reader_options const& options);

// API # 2
[[nodiscard]] std::vector<size_type> get_valid_row_groups(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::io::parquet_reader_options const& options);

// API # 3
[[nodiscard]] std::vector<size_type> filter_row_groups_with_stats(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream);

// API # 4
[[nodiscard]] std::pair<std::vector<cudf::io::text::byte_range_info>,
                        std::vector<cudf::io::text::byte_range_info>>
get_secondary_filters(std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
                      cudf::host_span<size_type const> row_group_indices,
                      cudf::io::parquet_reader_options const& options);

// API # 5
[[nodiscard]] std::vector<size_type> filter_row_groups_with_dictionary_pages(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  std::vector<rmm::device_buffer>& dictionary_page_data,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream);

// API # 6
[[nodiscard]] std::vector<size_type> filter_row_groups_with_bloom_filters(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  std::vector<rmm::device_buffer>& bloom_filter_data,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream);

// API # 7
[[nodiscard]] std::vector<std::vector<bool>> filter_data_pages_with_stats(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::host_span<size_type const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream);

// API # 8
[[nodiscard]] cudf::io::table_with_metadata materialize_filter_columns(
  std::unique_ptr<parquet::hybrid_scan_reader> const& reader,
  cudf::mutable_column_view input_rows,
  cudf::host_span<size_type const> row_group_indices,
  std::vector<rmm::device_buffer>& data_pages_bytes,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream);

}  // namespace experimental::io
}  // namespace CUDF_EXPORT cudf
