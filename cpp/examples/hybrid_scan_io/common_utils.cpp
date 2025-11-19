/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <string>
#include <vector>

/**
 * @file common_utils.cpp
 * @brief Definitions for utilities for `hybrid_scan_io` example
 */

std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(bool is_pool_used)
{
  if (is_pool_used) {
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      std::make_shared<rmm::mr::cuda_memory_resource>(), rmm::percent_of_free_device_memory(50));
  }
  return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

cudf::ast::operation create_filter_expression(std::string const& column_name,
                                              std::string const& literal_value)
{
  auto const column_reference = cudf::ast::column_name_reference(column_name);
  auto scalar                 = cudf::string_scalar(literal_value);
  auto literal                = cudf::ast::literal(scalar);
  return cudf::ast::operation(cudf::ast::ast_operator::EQUAL, column_reference, literal);
}

std::unique_ptr<cudf::table> combine_tables(std::unique_ptr<cudf::table> filter_table,
                                            std::unique_ptr<cudf::table> payload_table)
{
  auto filter_columns  = filter_table->release();
  auto payload_columns = payload_table->release();

  auto all_columns = std::vector<std::unique_ptr<cudf::column>>{};
  all_columns.reserve(filter_columns.size() + payload_columns.size());
  std::move(filter_columns.begin(), filter_columns.end(), std::back_inserter(all_columns));
  std::move(payload_columns.begin(), payload_columns.end(), std::back_inserter(all_columns));
  auto table = std::make_unique<cudf::table>(std::move(all_columns));

  return table;
}

void check_tables_equal(cudf::table_view const& lhs_table,
                        cudf::table_view const& rhs_table,
                        rmm::cuda_stream_view stream)
{
  try {
    // Left anti-join the original and transcoded tables identical tables should not throw an
    // exception and return an empty indices vector
    cudf::filtered_join join_obj(
      lhs_table, cudf::null_equality::EQUAL, cudf::set_as_build_table::RIGHT, stream);
    auto const indices = join_obj.anti_join(rhs_table, stream);
    // No exception thrown, check indices
    auto const tables_equal = indices->size() == 0;
    if (tables_equal) {
      std::cout << "Tables identical: " << std::boolalpha << tables_equal << "\n\n";
    } else {
      // Helper to write parquet data for inspection
      auto const write_parquet =
        [](cudf::table_view table, std::string filepath, rmm::cuda_stream_view stream) {
          auto sink_info = cudf::io::sink_info(filepath);
          auto opts      = cudf::io::parquet_writer_options::builder(sink_info, table).build();
          cudf::io::write_parquet(opts, stream);
        };
      write_parquet(lhs_table, "lhs_table.parquet", stream);
      write_parquet(rhs_table, "rhs_table.parquet", stream);
      throw std::logic_error("Tables identical: false\n\n");
    }
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }
}

cudf::host_span<uint8_t const> fetch_footer_bytes(cudf::host_span<uint8_t const> buffer)
{
  CUDF_FUNC_RANGE();

  using namespace cudf::io::parquet;

  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);
  size_t const len          = buffer.size();

  auto const header_buffer = cudf::host_span<uint8_t const>(buffer.data(), header_len);
  auto const header        = reinterpret_cast<file_header_s const*>(header_buffer.data());
  auto const ender_buffer =
    cudf::host_span<uint8_t const>(buffer.data() + len - ender_len, ender_len);
  auto const ender = reinterpret_cast<file_ender_s const*>(ender_buffer.data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  constexpr uint32_t parquet_magic = (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
  CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
               "Corrupted header or footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  return cudf::host_span<uint8_t const>(buffer.data() + len - ender->footer_len - ender_len,
                                        ender->footer_len);
}

cudf::host_span<uint8_t const> fetch_page_index_bytes(
  cudf::host_span<uint8_t const> buffer, cudf::io::text::byte_range_info const page_index_bytes)
{
  return cudf::host_span<uint8_t const>(
    reinterpret_cast<uint8_t const*>(buffer.data()) + page_index_bytes.offset(),
    page_index_bytes.size());
}

std::vector<rmm::device_buffer> fetch_byte_ranges(
  cudf::host_span<uint8_t const> host_buffer,
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  std::vector<rmm::device_buffer> buffers(byte_ranges.size());

  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(byte_ranges.size()),
                [&](auto const idx) {
                  auto const chunk_offset = host_buffer.data() + byte_ranges[idx].offset();
                  auto const chunk_size   = byte_ranges[idx].size();
                  auto buffer             = rmm::device_buffer(chunk_size, stream, mr);
                  CUDF_CUDA_TRY(cudaMemcpyAsync(
                    buffer.data(), chunk_offset, chunk_size, cudaMemcpyDefault, stream.value()));
                  buffers[idx] = std::move(buffer);
                });

  stream.synchronize_no_throw();
  return buffers;
}
