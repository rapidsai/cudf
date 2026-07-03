/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/common.hpp"
#include "io/parquet/parquet_common.hpp"
#include "io/parquet/reader_impl_helpers.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file parquet_io_utils.cpp
 * @brief Definitions for IO utilities for the Parquet and hybrid scan readers
 */

namespace cudf::io::parquet {

namespace {

/**
 * @brief Dispatches each indexed task and collects the results
 *
 * Dispatches sequentially or using host worker pool depending on the number of tasks.
 *
 * @tparam Task Callable invocable as `task(std::size_t task_idx)`
 * @param num_tasks Number of tasks to dispatch
 * @param task Task to run for each task index
 * @return Vector of results, one per task, in task-index order
 */
template <typename Task>
auto dispatch_tasks(std::size_t num_tasks, Task task)
{
  using result_type = std::invoke_result_t<Task, std::size_t>;

  auto constexpr parallel_threshold = 32;

  std::vector<result_type> results;
  results.reserve(num_tasks);

  if (num_tasks < parallel_threshold) {
    // Run sequentially to avoid task dispatch overhead
    std::for_each(cuda::counting_iterator<std::size_t>(0),
                  cuda::counting_iterator<std::size_t>(num_tasks),
                  [&](std::size_t task_idx) { results.emplace_back(task(task_idx)); });
  } else {
    // Dispatch the tasks to the host worker pool
    std::vector<std::future<result_type>> futures;
    futures.reserve(num_tasks);
    std::for_each(cuda::counting_iterator<std::size_t>(0),
                  cuda::counting_iterator<std::size_t>(num_tasks),
                  [&](std::size_t task_idx) {
                    futures.emplace_back(cudf::detail::host_worker_pool().submit_task(
                      [&task, task_idx]() { return task(task_idx); }));
                  });
    std::transform(futures.begin(), futures.end(), std::back_inserter(results), [](auto& fut) {
      return fut.get();
    });
  }
  return results;
}

/**
 * @copydoc cudf::io::parquet::fetch_footers_to_host
 */
std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_footers_to_host_impl(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources)
{
  // Look up runtime configuration once, as late as possible.
  auto const metadata_size_hint = cudf::io::parquet::metadata_size_hint();
  // Helper to fetch footer from a datasource
  auto const fetch_footer = [metadata_size_hint](cudf::io::datasource& datasource) {
    constexpr auto header_len = sizeof(file_header_s);
    constexpr auto ender_len  = sizeof(file_ender_s);
    size_t const len          = datasource.size();
    CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");

    auto const speculative_read_size =
      std::min(len, std::max(metadata_size_hint, static_cast<size_t>(ender_len)));
    auto const speculative_read_offset = len - speculative_read_size;

    auto speculative_buffer = datasource.host_read(speculative_read_offset, speculative_read_size);
    CUDF_EXPECTS(speculative_buffer->size() == speculative_read_size,
                 "Failed to read Parquet speculative metadata bytes");

    auto const ender = reinterpret_cast<file_ender_s const*>(
      speculative_buffer->data() + speculative_buffer->size() - ender_len);

    if (speculative_read_offset == 0) {
      auto const header = reinterpret_cast<file_header_s const*>(speculative_buffer->data());
      CUDF_EXPECTS(header->magic == detail::parquet_magic, "Corrupted header");
    }

    CUDF_EXPECTS(ender->magic == detail::parquet_magic, "Corrupted footer");
    CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
                 "Incorrect footer length");

    auto const footer_offset = len - ender->footer_len - ender_len;
    if (footer_offset >= speculative_read_offset) {
      // fastpath: the speculative read includes the full footer.
      auto const footer_start_offset = footer_offset - speculative_read_offset;
      CUDF_EXPECTS(footer_start_offset + ender->footer_len <= speculative_buffer->size(),
                   "Speculative metadata read did not include full footer bytes");
      std::vector<uint8_t> footer_bytes(ender->footer_len);
      std::memcpy(
        footer_bytes.data(), speculative_buffer->data() + footer_start_offset, ender->footer_len);
      return cudf::io::datasource::buffer::create(std::move(footer_bytes));
    }

    // The speculative read only got part of the footer. Read the missing prefix, then stitch.
    auto const missing_prefix_size = speculative_read_offset - footer_offset;
    auto missing_prefix            = datasource.host_read(footer_offset, missing_prefix_size);
    CUDF_EXPECTS(missing_prefix->size() == missing_prefix_size,
                 "Failed to read the missing footer prefix bytes");
    std::vector<uint8_t> footer_bytes(ender->footer_len);
    std::memcpy(footer_bytes.data(), missing_prefix->data(), missing_prefix_size);
    auto const footer_suffix_size = ender->footer_len - missing_prefix_size;
    std::memcpy(
      footer_bytes.data() + missing_prefix_size, speculative_buffer->data(), footer_suffix_size);
    return cudf::io::datasource::buffer::create(std::move(footer_bytes));
  };

  return dispatch_tasks(datasources.size(), [&](std::size_t source_idx) {
    return fetch_footer(datasources[source_idx].get());
  });
}

/**
 * @copydoc cudf::io::parquet::fetch_page_indexes_to_host
 */
std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_page_indexes_to_host_impl(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::io::text::byte_range_info const> page_index_bytes_per_source)
{
  CUDF_EXPECTS(datasources.size() == page_index_bytes_per_source.size(),
               "Encountered mismatch in number of datasources and page index byte ranges");

  // Helper to fetch page index bytes from a datasource
  auto const fetch_page_index = [](cudf::io::datasource& datasource,
                                   cudf::io::text::byte_range_info const& page_index_bytes) {
    CUDF_EXPECTS(
      page_index_bytes.offset() >= 0 and
        std::cmp_less_equal(page_index_bytes.offset() + page_index_bytes.size(), datasource.size()),
      std::format("Invalid page index byte range: offset={}, size={}, datasource_size={}",
                  page_index_bytes.offset(),
                  page_index_bytes.size(),
                  datasource.size()),
      std::out_of_range);
    return datasource.host_read(page_index_bytes.offset(), page_index_bytes.size());
  };

  return dispatch_tasks(datasources.size(), [&](std::size_t source_idx) {
    return fetch_page_index(datasources[source_idx].get(), page_index_bytes_per_source[source_idx]);
  });
}

using device_spans_per_source_type = std::vector<cudf::device_span<uint8_t const>>;

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<device_spans_per_source_type>,
           std::future<void>>
fetch_byte_ranges_to_device_async_impl(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::host_span<cudf::io::text::byte_range_info const> const>
    byte_ranges_per_source,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  static std::mutex host_read_mutex;
  static std::mutex device_read_mutex;

  auto const num_sources = datasources.size();

  CUDF_EXPECTS(num_sources == byte_ranges_per_source.size(),
               "Encountered mismatch in number of datasources and the number of byte range spans");

  // Total number of byte ranges across all sources
  auto const total_byte_ranges =
    std::accumulate(byte_ranges_per_source.begin(),
                    byte_ranges_per_source.end(),
                    std::size_t{0},
                    [](auto acc, auto const& ranges) { return acc + ranges.size(); });

  // IO descriptors
  std::vector<size_t> io_source_indices;
  std::vector<size_t> io_offsets;
  std::vector<size_t> io_sizes;
  std::vector<uint8_t*> destinations;
  io_source_indices.reserve(total_byte_ranges);
  io_offsets.reserve(total_byte_ranges);
  io_sizes.reserve(total_byte_ranges);
  destinations.reserve(total_byte_ranges);

  // Allocate one device buffer per byte ranges of a datasource
  std::vector<rmm::device_buffer> column_chunk_buffers{};
  column_chunk_buffers.reserve(num_sources);

  // Column chunk device spans, one per byte range per datasource
  std::vector<device_spans_per_source_type> column_chunk_data_per_source(num_sources);

  std::for_each(
    cuda::counting_iterator<cudf::size_type>(0),
    cuda::counting_iterator<cudf::size_type>(num_sources),
    [&](auto const source_idx) {
      auto const& byte_ranges = byte_ranges_per_source[source_idx];

      // Total buffer size required for column chunks of this source
      auto const buffer_size = std::accumulate(
        byte_ranges.begin(), byte_ranges.end(), std::size_t{0}, [](auto acc, auto const& range) {
          return acc + range.size();
        });

      // Buffer needs to be padded. Required by `gpuDecodePageData`.
      column_chunk_buffers.emplace_back(
        cudf::util::round_up_safe(buffer_size, cudf::io::detail::BUFFER_PADDING_MULTIPLE),
        stream,
        mr);

      auto buffer_data = static_cast<uint8_t*>(column_chunk_buffers.back().data());

      // Build device spans for each byte range in this source
      auto& column_chunk_data = column_chunk_data_per_source[source_idx];
      column_chunk_data.reserve(byte_ranges.size());
      std::ignore = std::accumulate(
        byte_ranges.begin(), byte_ranges.end(), std::size_t{0}, [&](auto acc, auto const& range) {
          column_chunk_data.emplace_back(buffer_data + acc, static_cast<size_t>(range.size()));
          return acc + range.size();
        });

      // Coalesce contiguous byte ranges within this source into single IO request
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
          io_source_indices.push_back(source_idx);
          io_offsets.push_back(io_offset);
          io_sizes.push_back(io_size);
          destinations.push_back(const_cast<uint8_t*>(column_chunk_data[chunk].data()));
        }
        chunk = next_chunk;
      }
    });

  CUDF_EXPECTS(io_offsets.size() == io_sizes.size() and io_sizes.size() == destinations.size() and
                 io_source_indices.size() == io_offsets.size(),
               "Unexpected number of IO source indices, offsets, sizes, or destinations");

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

  auto iter = cuda::make_zip_iterator(
    io_source_indices.begin(), io_offsets.begin(), io_sizes.begin(), destinations.begin());

  // Schedule host reads holding the `host_read_mutex` so that all reads for a caller thread
  // are scheduled without interleaving with reads from other threads yielding better pipelining
  {
    std::scoped_lock<std::mutex> lock(host_read_mutex);

    std::for_each(iter, iter + io_offsets.size(), [&](auto const& tuple) {
      auto const src_idx   = cuda::std::get<0>(tuple);
      auto const io_offset = cuda::std::get<1>(tuple);
      auto const io_size   = cuda::std::get<2>(tuple);
      auto const dest      = cuda::std::get<3>(tuple);

      auto& datasource = datasources[src_idx].get();
      if (not datasource.is_device_read_preferred(io_size)) {
        // Asynchronously read column chunk data to a host buffer
        host_read_tasks.emplace_back(cudf::detail::host_worker_pool().submit_task(
          [&datasource, io_offset, io_size]() -> host_read_buffer {
            return datasource.host_read(io_offset, io_size);
          }));
        copy_dsts.push_back(static_cast<void*>(dest));
        copy_sizes.push_back(io_size);
      }
    });
  }

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

  // Schedule device reads holding the `device_read_mutex` so that all reads for a caller thread
  // are scheduled without interleaving with reads from other threads yielding better pipelining
  {
    std::scoped_lock<std::mutex> lock(device_read_mutex);

    std::for_each(iter, iter + io_offsets.size(), [&](auto const& tuple) {
      auto const src_idx   = cuda::std::get<0>(tuple);
      auto const io_offset = cuda::std::get<1>(tuple);
      auto const io_size   = cuda::std::get<2>(tuple);
      auto const dest      = cuda::std::get<3>(tuple);

      auto& datasource = datasources[src_idx].get();
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
          std::move(column_chunk_data_per_source),
          std::async(std::launch::deferred, sync_function, std::move(device_read_tasks))};
}

std::pair<std::vector<rmm::device_buffer>, std::vector<device_spans_per_source_type>>
fetch_bloom_filters_to_device_impl(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::host_span<cudf::io::text::byte_range_info const> const>
    bloom_filter_byte_ranges_per_source,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref aligned_mr)
{
  auto const num_sources = datasources.size();
  CUDF_EXPECTS(num_sources == bloom_filter_byte_ranges_per_source.size(),
               "Encountered mismatch in number of datasources and bloom filter byte range spans");

  // One task per bloom filter, flattened across sources
  struct bloom_filter_task {
    std::size_t source_idx;
    std::size_t inner_idx;
    cudf::io::text::byte_range_info bloom_range;
    std::size_t host_offset{};
    std::size_t device_offset{};
    std::size_t header_size{};
    std::size_t bitset_size{};
    bool covered{};  // the whole bitset is already present in the source's host buffer
    bool present{};  // false for a chunk whose column has no bloom filter
  };

  // Phase 1: read bloom filters to host and parse their headers
  std::vector<bloom_filter_task> bloom_filter_tasks;
  bloom_filter_tasks.reserve(
    std::accumulate(bloom_filter_byte_ranges_per_source.begin(),
                    bloom_filter_byte_ranges_per_source.end(),
                    std::size_t{0},
                    [](auto acc, auto const& bloom_ranges) { return acc + bloom_ranges.size(); }));

  // Flatten per-source ranges while assigning each filter a task slot
  std::vector<std::size_t> per_source_host_size(num_sources, 0);
  std::for_each(cuda::counting_iterator<std::size_t>(0),
                cuda::counting_iterator<std::size_t>(num_sources),
                [&](std::size_t source_idx) {
                  auto const& bloom_ranges = bloom_filter_byte_ranges_per_source[source_idx];
                  std::for_each(cuda::counting_iterator<std::size_t>(0),
                                cuda::counting_iterator<std::size_t>(bloom_ranges.size()),
                                [&](std::size_t inner_idx) {
                                  auto const& bloom_range = bloom_ranges[inner_idx];
                                  auto const host_offset  = per_source_host_size[source_idx];
                                  auto const read_size =
                                    static_cast<std::size_t>(bloom_range.size());
                                  bloom_filter_tasks.push_back(
                                    {source_idx, inner_idx, bloom_range, host_offset});
                                  per_source_host_size[source_idx] = host_offset + read_size;
                                });
                });

  // Allocate one host buffer per source
  std::vector<std::vector<uint8_t>> host_buffers(num_sources);
  std::transform(per_source_host_size.begin(),
                 per_source_host_size.end(),
                 host_buffers.begin(),
                 [](std::size_t size) { return std::vector<uint8_t>(size); });

  // Speculatively read to host buffer & parse header
  auto const read_speculative = [&](std::size_t task_idx) -> bloom_filter_task {
    auto task = bloom_filter_tasks[task_idx];
    if (task.bloom_range.is_empty()) { return task; }
    auto const read_size = static_cast<std::size_t>(task.bloom_range.size());
    auto const host_slot =
      std::span<uint8_t>{host_buffers[task.source_idx]}.subspan(task.host_offset, read_size);
    auto const bytes_read = datasources[task.source_idx].get().host_read(
      static_cast<std::size_t>(task.bloom_range.offset()), read_size, host_slot.data());
    auto const header_info = detail::parse_bloom_filter_header({host_slot.data(), bytes_read});
    CUDF_EXPECTS(header_info.has_value(), "Encountered an invalid bloom filter header");
    auto const [header_bytes, bitset_size] = header_info.value();
    task.header_size                       = static_cast<std::size_t>(header_bytes);
    task.bitset_size                       = bitset_size;
    task.covered                           = bytes_read >= task.header_size + bitset_size;
    task.present                           = true;
    return task;
  };
  bloom_filter_tasks = dispatch_tasks(bloom_filter_tasks.size(), read_speculative);

  // Phase 2: allocate one aligned device buffer per source
  auto constexpr bitset_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;

  std::vector<std::size_t> per_source_device_size(num_sources, 0);
  std::for_each(bloom_filter_tasks.begin(), bloom_filter_tasks.end(), [&](bloom_filter_task& task) {
    if (not task.present) { return; }
    task.device_offset = per_source_device_size[task.source_idx];
    per_source_device_size[task.source_idx] +=
      cudf::util::round_up_safe(task.bitset_size, bitset_alignment);
  });

  std::vector<rmm::device_buffer> bloom_filter_buffers;
  bloom_filter_buffers.reserve(num_sources);
  std::transform(per_source_device_size.begin(),
                 per_source_device_size.end(),
                 std::back_inserter(bloom_filter_buffers),
                 [&](std::size_t size) { return rmm::device_buffer(size, stream, aligned_mr); });

  // Phase 3: make all bitsets host-resident, then batch-copy them to device
  std::vector<void*> copy_dsts;
  std::vector<void const*> copy_srcs;
  std::vector<std::size_t> copy_sizes;
  copy_dsts.reserve(bloom_filter_tasks.size());
  copy_srcs.reserve(bloom_filter_tasks.size());
  copy_sizes.reserve(bloom_filter_tasks.size());
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> bounce_buffers;

  // Allocate device span table
  std::vector<device_spans_per_source_type> bitset_spans_per_source(num_sources);
  std::transform(
    bloom_filter_byte_ranges_per_source.begin(),
    bloom_filter_byte_ranges_per_source.end(),
    bitset_spans_per_source.begin(),
    [](auto const& bloom_ranges) { return device_spans_per_source_type(bloom_ranges.size()); });

  std::for_each(
    bloom_filter_tasks.begin(), bloom_filter_tasks.end(), [&](bloom_filter_task const& task) {
      // Empty bloom filter: leave an empty span in place
      if (not task.present) { return; }
      // Aligned destination slot within this source's device buffer.
      auto& dst_buffer = bloom_filter_buffers[task.source_idx];
      auto const dst =
        cudf::device_span<uint8_t>{static_cast<uint8_t*>(dst_buffer.data()), dst_buffer.size()}
          .subspan(task.device_offset, task.bitset_size);
      CUDF_EXPECTS(reinterpret_cast<std::uintptr_t>(dst.data()) % bitset_alignment == 0,
                   "Encountered a misaligned bloom filter bitset");
      if (task.covered) {
        // Bitset fully covered: direct copy to device
        auto const src = std::span<uint8_t const>{host_buffers[task.source_idx]}.subspan(
          task.host_offset + task.header_size, task.bitset_size);
        copy_srcs.push_back(src.data());
      } else {
        // Bitset not fully covered: read from source, then copy to device
        auto const bitset_offset =
          static_cast<std::size_t>(task.bloom_range.offset()) + task.header_size;
        bounce_buffers.emplace_back(
          datasources[task.source_idx].get().host_read(bitset_offset, task.bitset_size));
        copy_srcs.push_back(bounce_buffers.back()->data());
      }
      copy_dsts.push_back(dst.data());
      copy_sizes.push_back(task.bitset_size);
      bitset_spans_per_source[task.source_idx][task.inner_idx] = dst;
    });

  // One batched host-to-device copy for every bitset
  if (not copy_dsts.empty()) {
    CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
      copy_dsts.data(), copy_srcs.data(), copy_sizes.data(), copy_dsts.size(), stream));
    stream.synchronize();
  }
  return {std::move(bloom_filter_buffers), std::move(bitset_spans_per_source)};
}

}  // namespace

[[nodiscard]] std::size_t metadata_size_hint()
{
  static constexpr auto default_metadata_size_hint = std::size_t{64} * 1024;
  return cudf::detail::getenv_or<std::size_t>("LIBCUDF_PARQUET_METADATA_SIZE_HINT",
                                              default_metadata_size_hint);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host(cudf::io::datasource& datasource)
{
  CUDF_FUNC_RANGE();
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  auto footer_buffers = fetch_footers_to_host_impl({datasources.data(), datasources.size()});
  return std::move(footer_buffers.front());
}

std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_footers_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources)
{
  CUDF_FUNC_RANGE();
  return fetch_footers_to_host_impl(datasources);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_to_host(
  cudf::io::datasource& datasource, cudf::io::text::byte_range_info const page_index_bytes)
{
  CUDF_FUNC_RANGE();

  // Wrap the inputs into arrays and delegate to the multi-source implementation
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  std::array<cudf::io::text::byte_range_info, 1> page_index_bytes_per_source{page_index_bytes};

  auto page_index_buffers = fetch_page_indexes_to_host_impl(
    {datasources.data(), datasources.size()},
    {page_index_bytes_per_source.data(), page_index_bytes_per_source.size()});
  return std::move(page_index_buffers.front());
}

std::vector<std::unique_ptr<cudf::io::datasource::buffer>> fetch_page_indexes_to_host(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<cudf::io::text::byte_range_info const> page_index_bytes_per_source)
{
  CUDF_FUNC_RANGE();
  return fetch_page_indexes_to_host_impl(datasources, page_index_bytes_per_source);
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<cudf::device_span<uint8_t const>>,
           std::future<void>>
fetch_byte_ranges_to_device_async(cudf::io::datasource& datasource,
                                  std::span<cudf::io::text::byte_range_info const> byte_ranges,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Wrap the inputs into arrays and delegate to the multi-source implementation
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  std::array<cudf::host_span<cudf::io::text::byte_range_info const>, 1> byte_ranges_per_source{
    cudf::host_span<cudf::io::text::byte_range_info const>{byte_ranges.data(), byte_ranges.size()}};

  auto [buffers, fetched_byte_ranges, fut] = fetch_byte_ranges_to_device_async_impl(
    {datasources.data(), datasources.size()},
    {byte_ranges_per_source.data(), byte_ranges_per_source.size()},
    stream,
    mr);

  return {std::move(buffers), std::move(fetched_byte_ranges.front()), std::move(fut)};
}

std::tuple<std::vector<rmm::device_buffer>,
           std::vector<std::vector<cudf::device_span<uint8_t const>>>,
           std::future<void>>
fetch_byte_ranges_to_device_async(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<std::vector<cudf::io::text::byte_range_info> const> byte_ranges_per_source,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Convert input vectors into host spans for the implementation
  std::vector<cudf::host_span<cudf::io::text::byte_range_info const>> byte_range_spans_per_source;
  byte_range_spans_per_source.reserve(byte_ranges_per_source.size());
  for (auto const& ranges : byte_ranges_per_source) {
    byte_range_spans_per_source.emplace_back(ranges);
  }
  return fetch_byte_ranges_to_device_async_impl(
    datasources,
    {byte_range_spans_per_source.data(), byte_range_spans_per_source.size()},
    stream,
    mr);
}

std::pair<std::vector<rmm::device_buffer>, std::vector<cudf::device_span<uint8_t const>>>
fetch_bloom_filters_to_device(
  cudf::io::datasource& datasource,
  cudf::host_span<cudf::io::text::byte_range_info const> bloom_filter_byte_ranges,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref aligned_mr)
{
  CUDF_FUNC_RANGE();

  // Wrap the inputs into arrays and delegate to the multi-source implementation
  std::array<std::reference_wrapper<cudf::io::datasource>, 1> datasources{std::ref(datasource)};
  std::array<cudf::host_span<cudf::io::text::byte_range_info const>, 1>
    bloom_filter_byte_ranges_per_source{bloom_filter_byte_ranges};

  auto [buffers, fetched_byte_ranges] = fetch_bloom_filters_to_device_impl(
    {datasources.data(), datasources.size()},
    {bloom_filter_byte_ranges_per_source.data(), bloom_filter_byte_ranges_per_source.size()},
    stream,
    aligned_mr);

  return {std::move(buffers), std::move(fetched_byte_ranges.front())};
}

std::pair<std::vector<rmm::device_buffer>,
          std::vector<std::vector<cudf::device_span<uint8_t const>>>>
fetch_bloom_filters_to_device(
  cudf::host_span<std::reference_wrapper<cudf::io::datasource> const> datasources,
  cudf::host_span<std::vector<cudf::io::text::byte_range_info> const>
    bloom_filter_byte_ranges_per_source,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref aligned_mr)
{
  CUDF_FUNC_RANGE();

  // Convert input vectors into host spans for the implementation
  std::vector<cudf::host_span<cudf::io::text::byte_range_info const>>
    bloom_filter_byte_range_spans_per_source;
  bloom_filter_byte_range_spans_per_source.reserve(bloom_filter_byte_ranges_per_source.size());
  for (auto const& ranges : bloom_filter_byte_ranges_per_source) {
    bloom_filter_byte_range_spans_per_source.emplace_back(ranges);
  }
  return fetch_bloom_filters_to_device_impl(datasources,
                                            {bloom_filter_byte_range_spans_per_source.data(),
                                             bloom_filter_byte_range_spans_per_source.size()},
                                            stream,
                                            aligned_mr);
}

}  // namespace cudf::io::parquet
