/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include "host_buffer_source.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <array>
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

io_backend::io_backend(cudf::host_span<std::byte const> buffer, rmm::cuda_stream_view stream)
  : _host_buffer_source(std::make_unique<host_buffer_source>(buffer)), _stream(stream)
{
  _datasource = cudf::io::datasource::create(_host_buffer_source.get());
  CUDF_EXPECTS(_datasource != nullptr, "Failed to create datasource from buffer");
}

io_backend::io_backend(std::string const& filepath, rmm::cuda_stream_view stream)
  : _datasource(cudf::io::datasource::create(filepath)), _stream(stream)
{
  CUDF_EXPECTS(_datasource != nullptr, "Failed to create datasource for: " + filepath);
}

std::vector<uint8_t> io_backend::fetch_footer_bytes()
{
  CUDF_FUNC_RANGE();

  using namespace cudf::io::parquet;

  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);
  size_t const len          = _datasource->size();

  std::array<uint8_t, header_len> header_buffer;
  std::array<uint8_t, ender_len> ender_buffer;

  fetch_byte_range_to_host(0, header_len, header_buffer.data());
  auto const header = reinterpret_cast<file_header_s const*>(header_buffer.data());
  fetch_byte_range_to_host(len - ender_len, ender_len, ender_buffer.data());
  auto const ender = reinterpret_cast<file_ender_s const*>(ender_buffer.data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  constexpr uint32_t parquet_magic = (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
  CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
               "Corrupted header or footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  std::vector<uint8_t> footer(ender->footer_len);
  fetch_byte_range_to_host(len - ender->footer_len - ender_len, ender->footer_len, footer.data());
  return footer;
}

std::vector<uint8_t> io_backend::fetch_page_index_bytes(
  cudf::io::text::byte_range_info const page_index_bytes)
{
  CUDF_FUNC_RANGE();
  std::vector<uint8_t> page_index(page_index_bytes.size());
  _datasource->host_read(page_index_bytes.offset(), page_index_bytes.size(), page_index.data());
  return page_index;
}

void io_backend::fetch_byte_range_to_host(size_t offset, size_t size, uint8_t* dst)
{
  CUDF_FUNC_RANGE();
  auto num_bytes_read = _datasource->host_read(offset, size, dst);
  CUDF_EXPECTS(num_bytes_read == size, "Failed to read expected number of bytes");
}

std::vector<rmm::device_buffer> io_backend::fetch_byte_ranges_to_device(
  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (byte_ranges.empty()) { return {}; }

  std::vector<rmm::device_buffer> buffers;
  buffers.reserve(byte_ranges.size());

  std::vector<std::future<size_t>> futures;
  futures.reserve(byte_ranges.size());

  for (auto const& byte_range : byte_ranges) {
    buffers.emplace_back(byte_range.size(), stream, mr);

    if (byte_range.size() == 0) { continue; }

    auto future = _datasource->device_read_async(byte_range.offset(),
                                                 byte_range.size(),
                                                 static_cast<uint8_t*>(buffers.back().data()),
                                                 _stream);
    futures.push_back(std::move(future));
  }

  for (auto&& future : futures) {
    future.get();
  }

  return buffers;
}
