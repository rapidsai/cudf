/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/common.hpp"
#include "io/parquet/parquet_common.hpp"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <algorithm>

/**
 * @file parquet_io_utils.cpp
 * @brief Definitions for IO utilities for the Parquet and hybrid scan readers
 */

namespace cudf::io::parquet {

std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_to_host(cudf::io::datasource& datasource)
{
  constexpr auto header_len = sizeof(file_header_s);
  constexpr auto ender_len  = sizeof(file_ender_s);
  size_t const len          = datasource.size();

  auto header_buffer = datasource.host_read(0, header_len);
  auto const header  = reinterpret_cast<file_header_s const*>(header_buffer->data());
  auto ender_buffer  = datasource.host_read(len - ender_len, ender_len);
  auto const ender   = reinterpret_cast<file_ender_s const*>(ender_buffer->data());
  CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
  CUDF_EXPECTS(header->magic == detail::parquet_magic, "Corrupted header");
  CUDF_EXPECTS(ender->magic == detail::parquet_magic, "Corrupted footer");
  CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
               "Incorrect footer length");

  return datasource.host_read(len - ender->footer_len - ender_len, ender->footer_len);
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_to_host(
  cudf::io::datasource& datasource, cudf::io::text::byte_range_info const page_index_bytes)
{
  return datasource.host_read(page_index_bytes.offset(), page_index_bytes.size());
}

namespace {

/**
 * @brief Page size required by GPUDirectStorage for aligned IO.
 *
 * GDS requires source offset, destination address, and read size to all be multiples of this value.
 */
constexpr size_t GDS_PAGE_SIZE = 4096;

/**
 * @brief Describes a coalesced group of adjacent byte ranges and how to read them.
 *
 * Adjacent input byte ranges are merged into a single IO plan. When the datasource supports
 * device reads and the aligned range fits within the datasource, the plan is marked as
 * GDS-aligned so that the read can leverage GPUDirectStorage.
 */
struct io_plan {
  size_t src_offset;      ///< Original (unaligned) source offset of the coalesced range
  size_t src_size;        ///< Original (unaligned) total size of the coalesced range
  size_t first_chunk;     ///< Index of the first input byte range in this group
  size_t num_chunks;      ///< Number of input byte ranges in this group
  bool gds_aligned;       ///< Whether this plan uses GDS-aligned IO
  size_t aligned_offset;  ///< Page-aligned source offset (<= src_offset)
  size_t aligned_size;    ///< Page-aligned read size (multiple of GDS_PAGE_SIZE)
  size_t prefix_bytes;    ///< Padding before the real data: src_offset - aligned_offset
};

/**
 * @brief IO dispatch method for a single read operation.
 *
 * GDS_DEVICE: device_read_async with page-aligned parameters (enables GDS).
 * PLAIN_DEVICE: device_read_async without alignment guarantees.
 * HOST: host_read followed by cuda_memcpy_async to device.
 */
enum class io_method { GDS_DEVICE, PLAIN_DEVICE, HOST };

/**
 * @brief A single IO read operation to be issued against the datasource.
 */
struct io_op {
  size_t src_offset;  ///< Source offset in the datasource
  size_t read_size;   ///< Number of bytes to read
  uint8_t* dest;      ///< Destination pointer in device memory
  io_method method;   ///< How to perform the read
};

}  // namespace

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

  auto const datasource_size = datasource.size();
  auto const can_device_read = datasource.supports_device_read();

  std::vector<cudf::device_span<uint8_t const>> column_chunk_data(byte_ranges.size());

  // Phase 1: Coalesce adjacent byte ranges and build an IO plan for each.
  // For GDS-eligible reads we extend the range to page-aligned boundaries so that
  // device_read_async can leverage GPUDirectStorage.
  std::vector<io_plan> plans;
  size_t non_gds_total_size = 0;

  for (size_t chunk = 0; chunk < byte_ranges.size();) {
    auto const io_offset = static_cast<size_t>(byte_ranges[chunk].offset());
    auto io_size         = static_cast<size_t>(byte_ranges[chunk].size());
    size_t const first   = chunk;
    size_t next          = chunk + 1;
    while (next < byte_ranges.size()) {
      if (static_cast<size_t>(byte_ranges[next].offset()) != io_offset + io_size) { break; }
      io_size += byte_ranges[next].size();
      next++;
    }

    if (io_size != 0) {
      io_plan plan{};
      plan.src_offset  = io_offset;
      plan.src_size    = io_size;
      plan.first_chunk = first;
      plan.num_chunks  = next - first;
      plan.gds_aligned = false;

      // Try to make this a GDS-aligned read: align offset down and end up to GDS_PAGE_SIZE.
      // This is only possible when the rounded-up end doesn't exceed the datasource.
      if (can_device_read && datasource.is_device_read_preferred(io_size) &&
          datasource_size >= GDS_PAGE_SIZE) {
        auto const a_offset = cudf::util::round_down_safe(io_offset, GDS_PAGE_SIZE);
        auto const a_end    = cudf::util::round_up_safe(io_offset + io_size, GDS_PAGE_SIZE);

        if (a_end <= datasource_size) {
          plan.gds_aligned    = true;
          plan.aligned_offset = a_offset;
          plan.aligned_size   = a_end - a_offset;
          plan.prefix_bytes   = io_offset - a_offset;
        }
      }

      if (!plan.gds_aligned) { non_gds_total_size += io_size; }
      plans.push_back(plan);
    }
    chunk = next;
  }

  // Phase 2: Allocate device buffers.
  //  - One shared buffer for all non-GDS reads (packed contiguously, padded for the decoder).
  //  - One dedicated buffer per GDS read (over-allocated by GDS_PAGE_SIZE so we can always
  //    find a page-aligned destination address inside it).
  // rmm::device_buffer move-constructor preserves device pointers, so growing the vector
  // after capturing a pointer from an earlier element is safe.
  std::vector<rmm::device_buffer> buffers;
  auto const num_gds_plans = static_cast<size_t>(
    std::count_if(plans.begin(), plans.end(), [](auto const& p) { return p.gds_aligned; }));
  buffers.reserve((non_gds_total_size > 0 ? 1 : 0) + num_gds_plans);

  uint8_t* non_gds_base = nullptr;
  if (non_gds_total_size > 0) {
    buffers.emplace_back(
      cudf::util::round_up_safe(non_gds_total_size, cudf::io::detail::BUFFER_PADDING_MULTIPLE),
      stream,
      mr);
    non_gds_base = static_cast<uint8_t*>(buffers.back().data());
  }

  std::vector<io_op> io_ops;
  io_ops.reserve(plans.size());

  size_t non_gds_buf_offset = 0;

  for (auto const& plan : plans) {
    if (plan.gds_aligned) {
      buffers.emplace_back(plan.aligned_size + GDS_PAGE_SIZE, stream, mr);
      auto* buf_ptr = static_cast<uint8_t*>(buffers.back().data());

      // Find the first page-aligned address within the buffer by rounding up to GDS_PAGE_SIZE
      auto const addr               = reinterpret_cast<uintptr_t>(buf_ptr);
      auto constexpr alignment_mask = static_cast<uintptr_t>(GDS_PAGE_SIZE) - 1;
      auto* aligned_dest = reinterpret_cast<uint8_t*>((addr + alignment_mask) & ~alignment_mask);

      // The actual requested data starts at aligned_dest + prefix_bytes
      auto* data_start           = aligned_dest + plan.prefix_bytes;
      size_t offset_in_coalesced = 0;
      for (size_t i = plan.first_chunk; i < plan.first_chunk + plan.num_chunks; ++i) {
        auto const range_size = static_cast<size_t>(byte_ranges[i].size());
        column_chunk_data[i]  = {data_start + offset_in_coalesced, range_size};
        offset_in_coalesced += range_size;
      }

      io_ops.push_back(
        {plan.aligned_offset, plan.aligned_size, aligned_dest, io_method::GDS_DEVICE});
    } else {
      auto* dest                 = non_gds_base + non_gds_buf_offset;
      size_t offset_in_coalesced = 0;
      for (size_t i = plan.first_chunk; i < plan.first_chunk + plan.num_chunks; ++i) {
        auto const range_size = static_cast<size_t>(byte_ranges[i].size());
        column_chunk_data[i]  = {dest + offset_in_coalesced, range_size};
        offset_in_coalesced += range_size;
      }

      auto const method = (can_device_read && datasource.is_device_read_preferred(plan.src_size))
                            ? io_method::PLAIN_DEVICE
                            : io_method::HOST;
      io_ops.push_back({plan.src_offset, plan.src_size, dest, method});

      non_gds_buf_offset += plan.src_size;
    }
  }

  // Phase 3: Issue all IO reads.
  std::vector<std::future<size_t>> device_read_tasks;
  std::vector<std::future<size_t>> host_read_tasks;
  device_read_tasks.reserve(io_ops.size());
  host_read_tasks.reserve(io_ops.size());

  // device_read_async is not guaranteed to follow stream-ordering (see datasource API docs).
  stream.synchronize();

  {
    std::lock_guard<std::mutex> lock(mutex);

    for (auto const& op : io_ops) {
      switch (op.method) {
        case io_method::GDS_DEVICE:
        case io_method::PLAIN_DEVICE:
          device_read_tasks.emplace_back(
            datasource.device_read_async(op.src_offset, op.read_size, op.dest, stream));
          break;
        case io_method::HOST: {
          auto const offset = op.src_offset;
          auto const size   = op.read_size;
          auto* dest        = op.dest;
          host_read_tasks.emplace_back(
            std::async(std::launch::deferred, [&datasource, offset, size, dest, stream]() {
              auto host_buffer = datasource.host_read(offset, size);
              cudf::detail::cuda_memcpy_async(
                cudf::device_span<uint8_t>{dest, size},
                cudf::host_span<uint8_t const>{host_buffer->data(), size},
                stream);
              return size;
            }));
        } break;
      }
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
  return {std::move(buffers),
          std::move(column_chunk_data),
          std::async(std::launch::deferred,
                     sync_function,
                     std::move(host_read_tasks),
                     std::move(device_read_tasks))};
}

}  // namespace cudf::io::parquet
