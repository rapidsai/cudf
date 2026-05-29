/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/common.hpp"
#include "io/parquet/parquet_common.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <functional>
#include <mutex>
#include <numeric>
#include <tuple>

/**
 * @file parquet_io_utils.cpp
 * @brief Definitions for IO utilities for the Parquet and hybrid scan readers
 */

namespace cudf::io::parquet {

namespace detail {

auto constexpr parallel_threshold = 16;

std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_footers_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources)
{
  // Helper to fetch footer from a datasource
  auto const fetch_footer = [](cudf::io::datasource& datasource) {
    constexpr auto header_len = sizeof(file_header_s);
    constexpr auto ender_len  = sizeof(file_ender_s);
    size_t const len          = datasource.size();

    auto header_buffer = datasource.host_read(0, header_len);
    auto const header  = reinterpret_cast<file_header_s const*>(header_buffer->data());
    auto ender_buffer  = datasource.host_read(len - ender_len, ender_len);
    auto const ender   = reinterpret_cast<file_ender_s const*>(ender_buffer->data());
    CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
    CUDF_EXPECTS(header->magic == parquet_magic, "Corrupted header");
    CUDF_EXPECTS(ender->magic == parquet_magic, "Corrupted footer");
    CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
                 "Incorrect footer length");

    return datasource.host_read(len - ender->footer_len - ender_len, ender->footer_len);
  };

  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> footer_buffers;
  footer_buffers.reserve(datasources.size());
  auto const num_sources = datasources.size();

  if (num_sources < parallel_threshold) {
    // Read footers sequentially to avoid task dispatch overhead
    std::transform(datasources.begin(),
                   datasources.end(),
                   std::back_inserter(footer_buffers),
                   [&](auto const& datasource_ref) { return fetch_footer(datasource_ref.get()); });
  } else {
    // Read footers in parallel
    std::vector<std::future<std::unique_ptr<cudf::io::datasource::buffer>>> tasks;
    tasks.reserve(datasources.size());
    for (auto const& datasource_ref : datasources) {
      tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(
        [&datasource = datasource_ref.get(), &fetch_footer]() {
          return fetch_footer(datasource);
        }));
    }
    std::transform(tasks.begin(), tasks.end(), std::back_inserter(footer_buffers), [](auto& task) {
      return task.get();
    });
  }
  return footer_buffers;
}

std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_page_indexes_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::io::text::byte_range_info const> page_index_bytes_per_source)
{
  CUDF_EXPECTS(datasources.size() == page_index_bytes_per_source.size(),
               "Encountered mismatch in number of datasources and page index byte ranges");

  // Helper to fetch page index bytes from a datasource
  auto const fetch_page_index = [](cudf::io::datasource& datasource,
                                   cudf::io::text::byte_range_info const& page_index_bytes) {
    return datasource.host_read(page_index_bytes.offset(), page_index_bytes.size());
  };

  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> page_index_buffers;
  page_index_buffers.reserve(datasources.size());

  auto const num_sources = datasources.size();
  auto iter = cuda::make_zip_iterator(datasources.begin(), page_index_bytes_per_source.begin());

  if (num_sources < parallel_threshold) {
    // Read page indexes sequentially to avoid task dispatch overhead
    std::transform(
      iter, iter + datasources.size(), std::back_inserter(page_index_buffers), [&](auto const& t) {
        return fetch_page_index(cuda::std::get<0>(t).get(), cuda::std::get<1>(t));
      });
  } else {
    // Read page indexes in parallel
    std::vector<std::future<std::unique_ptr<cudf::io::datasource::buffer>>> tasks;
    tasks.reserve(datasources.size());
    std::for_each(iter, iter + datasources.size(), [&](auto const& tuple) {
      auto const& datasource       = cuda::std::get<0>(tuple);
      auto const& page_index_bytes = cuda::std::get<1>(tuple);
      tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(
        [&datasource = datasource.get(), page_index_bytes, &fetch_page_index]() {
          return fetch_page_index(datasource, page_index_bytes);
        }));
    });
    std::transform(
      tasks.begin(), tasks.end(), std::back_inserter(page_index_buffers), [](auto& task) {
        return task.get();
      });
  }
  return page_index_buffers;
}

}  // namespace detail

std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host(cudf::io::datasource& datasource)
{
  CUDF_FUNC_RANGE();

  // Wrap the input into an array and delegate to the detail multi-source API
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  auto footer_buffers = detail::fetch_footers_to_host({datasources.data(), datasources.size()});
  return std::move(footer_buffers.front());
}

std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_footers_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources)
{
  CUDF_FUNC_RANGE();
  return detail::fetch_footers_to_host(datasources);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_to_host(
  cudf::io::datasource& datasource, cudf::io::text::byte_range_info const page_index_bytes)
{
  CUDF_FUNC_RANGE();

  // Wrap the inputs into arrays and delegate to the detail multi-source API
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  std::array<cudf::io::text::byte_range_info, 1> page_index_bytes_per_source{page_index_bytes};

  auto page_index_buffers = detail::fetch_page_indexes_to_host(
    {datasources.data(), datasources.size()},
    {page_index_bytes_per_source.data(), page_index_bytes_per_source.size()});
  return std::move(page_index_buffers.front());
}

std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_page_indexes_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::io::text::byte_range_info const> page_index_bytes_per_source)
{
  CUDF_FUNC_RANGE();
  return detail::fetch_page_indexes_to_host(datasources, page_index_bytes_per_source);
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<cudf::device_span<uint8_t const>>,
           std::future<void>>
fetch_byte_ranges_to_device_async(
  cudf::io::datasource& datasource,
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
  // Buffer needs to be padded. Required by `gpuDecodePageData`.
  column_chunk_buffers.emplace_back(
    cudf::util::round_up_safe(total_size, cudf::io::detail::BUFFER_PADDING_MULTIPLE), stream, mr);
  auto buffer_data = static_cast<uint8_t*>(column_chunk_buffers.back().data());
  std::ignore      = std::accumulate(
    byte_ranges.begin(), byte_ranges.end(), std::size_t{0}, [&](auto acc, auto const& range) {
      column_chunk_data.emplace_back(buffer_data + acc, static_cast<size_t>(range.size()));
      return acc + range.size();
    });

  std::vector<size_t> io_offsets;
  std::vector<size_t> io_sizes;
  std::vector<uint8_t*> destinations;
  io_offsets.reserve(byte_ranges.size());
  io_sizes.reserve(byte_ranges.size());
  destinations.reserve(byte_ranges.size());

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
      io_offsets.push_back(io_offset);
      io_sizes.push_back(io_size);
      destinations.push_back(const_cast<uint8_t*>(column_chunk_data[chunk].data()));
    }
    chunk = next_chunk;
  }
  CUDF_EXPECTS(io_offsets.size() == io_sizes.size() and io_sizes.size() == destinations.size(),
               "Unexpected number of IO offsets, sizes, or destinations");

  using host_read_buffer = std::unique_ptr<cudf::io::datasource::buffer>;

  // Vectors to hold futures from datasource
  std::vector<std::future<size_t>> device_read_tasks{};
  std::vector<std::future<host_read_buffer>> host_read_tasks{};
  device_read_tasks.reserve(io_offsets.size());
  host_read_tasks.reserve(io_offsets.size());

  // Vectors to store intermediate host buffers and relevant pointers
  std::vector<host_read_buffer> host_buffers{};
  std::vector<void const*> copy_srcs{};
  std::vector<void*> copy_dsts{};
  std::vector<size_t> copy_sizes{};
  copy_dsts.reserve(io_offsets.size());
  copy_sizes.reserve(io_offsets.size());

  auto iter = cuda::make_zip_iterator(io_offsets.begin(), io_sizes.begin(), destinations.begin());

  // Schedule host reads in parallel
  std::for_each(iter, iter + io_offsets.size(), [&](auto const& tuple) {
    auto const io_offset = cuda::std::get<0>(tuple);
    auto const io_size   = cuda::std::get<1>(tuple);
    auto const dest      = cuda::std::get<2>(tuple);

    if (not datasource.is_device_read_preferred(io_size)) {
      // Asynchronously read column chunk data to a host buffer
      host_read_tasks.emplace_back(datasource.host_read_async(io_offset, io_size));
      copy_dsts.push_back(static_cast<void*>(dest));
      copy_sizes.push_back(io_size);
    }
  });

  // Complete host reads
  if (not host_read_tasks.empty()) {
    copy_srcs.reserve(host_read_tasks.size());
    host_buffers.reserve(host_read_tasks.size());

    for (auto& task : host_read_tasks) {
      host_buffers.emplace_back(task.get());
      copy_srcs.push_back(host_buffers.back().get()->data());
    }
  }

  // `device_read_async` is not guaranteed to follow stream-ordering (see datasource API docs)
  stream.synchronize();

  // Ensure all device reads for this thread are scheduled together
  {
    std::scoped_lock<std::mutex> lock(mutex);

    std::for_each(iter, iter + io_offsets.size(), [&](auto const& tuple) {
      auto const io_offset = cuda::std::get<0>(tuple);
      auto const io_size   = cuda::std::get<1>(tuple);
      auto const dest      = cuda::std::get<2>(tuple);

      // Directly read the column chunk data to the device buffer if supported
      if (datasource.is_device_read_preferred(io_size)) {
        device_read_tasks.emplace_back(
          datasource.device_read_async(io_offset, io_size, dest, stream));
      }
    });

    // Schedule a batched memcpy from host buffers to device
    if (not host_buffers.empty()) {
      CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
        copy_dsts.data(), copy_srcs.data(), copy_sizes.data(), copy_dsts.size(), stream));
    }
  }

  // Synchronize stream if `memcpy_batch_async` was called to safely discard the host buffers
  if (not host_buffers.empty()) { stream.synchronize(); }

  auto sync_function = [](decltype(device_read_tasks) device_read_tasks) {
    for (auto& task : device_read_tasks) {
      task.get();
    }
  };
  return {std::move(column_chunk_buffers),
          std::move(column_chunk_data),
          std::async(std::launch::deferred, sync_function, std::move(device_read_tasks))};
}

}  // namespace cudf::io::parquet
