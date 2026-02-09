/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <numeric>
#include <string>
#include <vector>

/**
 * @file common_utils.cpp
 * @brief Definitions for utilities for `hybrid_scan_io` example
 */

bool get_boolean(std::string input)
{
  std::transform(input.begin(), input.end(), input.begin(), ::toupper);

  // Check if the input string matches to any of the following
  return input == "ON" or input == "TRUE" or input == "YES" or input == "Y" or input == "T";
}

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

std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_bytes(cudf::io::datasource& datasource)
{
  CUDF_FUNC_RANGE();

  using namespace cudf::io::parquet;

  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);
  size_t const len          = datasource.size();

  auto header_buffer = datasource.host_read(0, header_len);
  auto const header  = reinterpret_cast<file_header_s const*>(header_buffer->data());
  auto ender_buffer  = datasource.host_read(len - ender_len, ender_len);
  auto const ender   = reinterpret_cast<file_ender_s const*>(ender_buffer->data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  constexpr uint32_t parquet_magic = (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
  CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
               "Corrupted header or footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  return datasource.host_read(len - ender->footer_len - ender_len, ender->footer_len);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_bytes(
  cudf::io::datasource& datasource, cudf::io::text::byte_range_info const page_index_bytes)
{
  return datasource.host_read(page_index_bytes.offset(), page_index_bytes.size());
}

cudf::host_span<uint8_t const> make_host_span(
  std::reference_wrapper<cudf::io::datasource::buffer const> buffer)
{
  return cudf::host_span<uint8_t const>{static_cast<uint8_t const*>(buffer.get().data()),
                                        buffer.get().size()};
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<cudf::device_span<uint8_t const>>,
           std::future<void>>
fetch_byte_ranges(cudf::io::datasource& datasource,
                  cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  static std::mutex mutex;

  // Allocate device spans for each column chunk
  std::vector<cudf::device_span<uint8_t const>> column_chunk_data{};
  column_chunk_data.reserve(byte_ranges.size());

  auto total_size = std::accumulate(
    byte_ranges.begin(), byte_ranges.end(), std::size_t{0}, [&](auto acc, auto const& range) {
      return acc + range.size();
    });

  // Allocate single device buffer for all column chunks
  std::vector<rmm::device_buffer> column_chunk_buffers{};
  column_chunk_buffers.emplace_back(total_size, stream, mr);
  auto buffer_data = static_cast<uint8_t*>(column_chunk_buffers.back().data());
  std::ignore      = std::accumulate(
    byte_ranges.begin(), byte_ranges.end(), std::size_t{0}, [&](auto acc, auto const& range) {
      column_chunk_data.emplace_back(buffer_data + acc, static_cast<size_t>(range.size()));
      return acc + range.size();
    });

  std::vector<std::future<size_t>> device_read_tasks{};
  std::vector<std::future<size_t>> host_read_tasks{};
  device_read_tasks.reserve(byte_ranges.size());
  host_read_tasks.reserve(byte_ranges.size());
  {
    std::lock_guard<std::mutex> lock(mutex);

    for (size_t chunk = 0; chunk < byte_ranges.size();) {
      auto const io_offset = static_cast<size_t>(byte_ranges[chunk].offset());
      auto io_size         = static_cast<size_t>(byte_ranges[chunk].size());
      size_t next_chunk    = chunk + 1;
      while (next_chunk < byte_ranges.size()) {
        size_t const next_offset = byte_ranges[next_chunk].offset();
        if (next_offset != io_offset + io_size) { break; }
        io_size += byte_ranges[next_chunk].size();
        next_chunk++;
      }

      if (io_size != 0) {
        auto dest = const_cast<uint8_t*>(column_chunk_data[chunk].data());
        // Directly read the column chunk data to the device
        // buffer if supported
        if (datasource.supports_device_read() and datasource.is_device_read_preferred(io_size)) {
          device_read_tasks.emplace_back(
            datasource.device_read_async(io_offset, io_size, dest, stream));
        } else {
          // Read the column chunk data to the host buffer and
          // copy it to the device buffer
          host_read_tasks.emplace_back(
            std::async(std::launch::deferred, [&datasource, io_offset, io_size, dest, stream]() {
              auto host_buffer = datasource.host_read(io_offset, io_size);
              cudf::detail::cuda_memcpy_async(
                cudf::device_span<uint8_t>{dest, io_size},
                cudf::host_span<uint8_t const>{host_buffer->data(), io_size},
                stream);
              return io_size;
            }));
        }
      }
      chunk = next_chunk;
    }
  }

  auto sync_function = [](decltype(host_read_tasks) host_read_tasks,
                          decltype(device_read_tasks) device_read_tasks) {
    for (auto& task : host_read_tasks) {
      task.get();
    }
    for (auto& task : device_read_tasks) {
      task.get();
    }
  };
  return {std::move(column_chunk_buffers),
          std::move(column_chunk_data),
          std::async(std::launch::deferred,
                     sync_function,
                     std::move(host_read_tasks),
                     std::move(device_read_tasks))};
}

std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>> tables,
                                                rmm::cuda_stream_view stream)
{
  if (tables.size() == 1) { return std::move(tables[0]); }

  std::vector<cudf::table_view> table_views;
  table_views.reserve(tables.size());
  std::transform(
    tables.begin(), tables.end(), std::back_inserter(table_views), [&](auto const& tbl) {
      return tbl->view();
    });
  // Construct the final table
  return cudf::concatenate(table_views, stream);
}
