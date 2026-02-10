/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/aligned_resource_adaptor.hpp>

auto constexpr bloom_filter_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;

/**
 * @brief Read parquet file with the hybrid scan reader
 *
 * @param datasource Input datasource
 * @param filter_expression Filter expression
 * @param num_filter_columns Number of filter columns
 * @param payload_column_names List of paths of select payload column names, if any
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 *
 * @return Tuple of filter table, payload table, and the final
 *         row validity column
 */
std::
  tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>>
  hybrid_scan(cudf::io::datasource& datasource,
              cudf::ast::operation const& filter_expression,
              cudf::size_type num_filter_columns,
              std::optional<std::vector<std::string>> const& payload_column_names,
              rmm::cuda_stream_view stream,
              rmm::device_async_resource_ref mr,
              rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>& aligned_mr);

/**
 * @brief Read parquet file with the hybrid scan reader
 *
 * @param datasource Input datasource
 * @param filter_expression Filter expression
 * @param num_filter_columns Number of filter columns
 * @param payload_column_names List of paths of select payload column names, if any
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 *
 * @return Tuple of filter table, payload table, and the final
 *         row validity column
 */
std::tuple<std::unique_ptr<cudf::table>,
           std::unique_ptr<cudf::table>,
           std::unique_ptr<cudf::column>>
chunked_hybrid_scan(cudf::io::datasource& datasource,
                    cudf::ast::operation const& filter_expression,
                    cudf::size_type num_filter_columns,
                    std::optional<std::vector<std::string>> const& payload_column_names,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr,
                    rmm::mr::aligned_resource_adaptor<rmm::mr::device_memory_resource>& aligned_mr);

/**
 * @brief Read parquet file with the hybrid scan reader in a single step
 *
 * @param datasource Input datasource
 * @param filter_expression Filter expression
 * @param column_names List of column names to read, if any
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Read table and metadata
 */
cudf::io::table_with_metadata hybrid_scan_single_step(
  cudf::io::datasource& datasource,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Read parquet file with the hybrid scan reader in a single step using chunked reading
 *
 * @param file_buffer_span Input parquet buffer span
 * @param filter_expression Filter expression
 * @param column_names List of column names to read, if any
 * @param stream CUDA stream
 * @param mr Device memory resource
 *
 * @return Read table and metadata
 */
cudf::io::table_with_metadata chunked_hybrid_scan_single_step(
  cudf::io::datasource& datasource,
  cudf::ast::operation const& filter_expression,
  std::optional<std::vector<std::string>> const& column_names,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
