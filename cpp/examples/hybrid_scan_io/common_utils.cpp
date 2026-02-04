/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include "timer.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/aligned_resource_adaptor.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

/**
 * @file common_utils.cpp
 * @brief Definitions for utilities for `hybrid_scan_io` example
 */

bool get_boolean(std::string input)
{
  std::transform(input.begin(), input.end(), input.begin(), ::toupper);
  return input == "ON" or input == "TRUE" or input == "YES" or input == "Y" or input == "T";
}

std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(bool is_pool_used)
{
  if (is_pool_used) {
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      std::make_shared<rmm::mr::cuda_memory_resource>(), rmm::percent_of_free_device_memory(80));
  }
  return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

std::vector<io_source> extract_input_sources(std::string const& paths,
                                             int32_t input_multiplier,
                                             int32_t thread_count,
                                             io_source_type io_source_type,
                                             rmm::cuda_stream_view stream)
{
  // Get the delimited paths to directory and/or files.
  std::vector<std::string> const delimited_paths = [&]() {
    std::vector<std::string> paths_list;
    std::stringstream strstream{paths};
    std::string path;
    // Extract the delimited paths.
    while (std::getline(strstream, path, char{','})) {
      paths_list.push_back(path);
    }
    return paths_list;
  }();

  // List of parquet files
  std::vector<std::string> parquet_files;
  std::for_each(delimited_paths.cbegin(), delimited_paths.cend(), [&](auto const& path_string) {
    auto const path = std::filesystem::path{path_string};
    // If this is a parquet file, add it.
    if (std::filesystem::is_regular_file(path)) {
      parquet_files.push_back(path_string);
    }
    // If this is a directory, add all files in the directory.
    else if (std::filesystem::is_directory(path)) {
      for (auto const& file : std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_regular_file(file.path())) {
          parquet_files.push_back(file.path().string());
        } else {
          std::cout << "Skipping sub-directory: " << file.path().string() << "\n";
        }
      }
    } else {
      throw std::runtime_error("Encountered an invalid input path\n");
    }
  });

  // Current size of list of parquet files
  auto const initial_size = parquet_files.size();
  if (initial_size == 0) { throw std::runtime_error("No input files to read. Exiting early.\n"); }

  // Reserve space
  parquet_files.reserve(std::max<size_t>(thread_count, input_multiplier * parquet_files.size()));

  // Append the input files by input_multiplier times
  std::for_each(thrust::make_counting_iterator(1),
                thrust::make_counting_iterator(input_multiplier),
                [&](auto i) {
                  parquet_files.insert(parquet_files.end(),
                                       parquet_files.begin(),
                                       parquet_files.begin() + initial_size);
                });

  if (parquet_files.size() < thread_count) {
    // Cycle append parquet files from the existing ones if less than the thread_count
    std::cout << "Warning: Number of input sources < thread count. Cycling from\n"
                 "and appending to current input sources such that the number of\n"
                 "input source == thread count\n";
    for (size_t idx = 0; thread_count > static_cast<int>(parquet_files.size()); idx++) {
      parquet_files.emplace_back(parquet_files[idx % initial_size]);
    }
  }

  // Vector of io sources
  std::vector<io_source> input_sources;
  input_sources.reserve(parquet_files.size());
  // Transform input files to the specified io sources
  std::transform(
    parquet_files.begin(),
    parquet_files.end(),
    std::back_inserter(input_sources),
    [&](auto const& file_name) { return io_source{file_name, io_source_type, stream}; });
  stream.synchronize();
  return input_sources;
}

cudf::ast::operation create_filter_expression(std::string const& column_name,
                                              std::string const& literal_value)
{
  auto const column_reference = cudf::ast::column_name_reference(column_name);
  auto scalar                 = cudf::string_scalar(literal_value);
  auto literal                = cudf::ast::literal(scalar);
  return cudf::ast::operation(cudf::ast::ast_operator::EQUAL, column_reference, literal);
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
  static std::mutex mutex;

  CUDF_FUNC_RANGE();

  static std::mutex mutex;

  std::vector<rmm::device_buffer> buffers(byte_ranges.size());
  {
    std::lock_guard<std::mutex> lock(mutex);

    std::transform(
      byte_ranges.begin(), byte_ranges.end(), buffers.begin(), [&](auto const& byte_range) {
        auto const chunk_offset = host_buffer.data() + byte_range.offset();
        auto const chunk_size   = static_cast<size_t>(byte_range.size());
        auto buffer             = rmm::device_buffer(chunk_size, stream, mr);
        cudf::detail::cuda_memcpy_async(
          cudf::device_span<uint8_t>{static_cast<uint8_t*>(buffer.data()), chunk_size},
          cudf::host_span<uint8_t const>{chunk_offset, chunk_size},
          stream);
        return buffer;
      });
  }

  return buffers;
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
