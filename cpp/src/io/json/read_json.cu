/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "io/comp/io_uncomp.hpp"
#include "io/json/legacy/read_json.hpp"
#include "io/json/nested_json.hpp"
#include "read_json.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/scatter.h>

#include <numeric>

namespace cudf::io::json::detail {

size_t sources_size(host_span<std::unique_ptr<datasource>> const sources,
                    size_t range_offset,
                    size_t range_size)
{
  return std::accumulate(sources.begin(), sources.end(), 0ul, [=](size_t sum, auto& source) {
    auto const size = source->size();
    // TODO take care of 0, 0, or *, 0 case.
    return sum +
           (range_size == 0 or range_offset + range_size > size ? size - range_offset : range_size);
  });
}

/**
 * @brief Read from array of data sources into RMM buffer
 *
 * @param sources Array of data sources
 * @param compression Compression format of source
 * @param range_offset Number of bytes to skip from source start
 * @param range_size Number of bytes to read from source
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
rmm::device_uvector<char> ingest_raw_input(host_span<std::unique_ptr<datasource>> sources,
                                           compression_type compression,
                                           size_t range_offset,
                                           size_t range_size,
                                           rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  // We append a line delimiter between two files to make sure the last line of file i and the first
  // line of file i+1 don't end up on the same JSON line, if file i does not already end with a line
  // delimiter.
  auto constexpr num_delimiter_chars = 1;
  auto const num_extra_delimiters    = num_delimiter_chars * (sources.size() - 1);

  // Iterate through the user defined sources and read the contents into the local buffer
  auto const total_source_size =
    sources_size(sources, range_offset, range_size) + num_extra_delimiters;

  if (compression == compression_type::NONE) {
    std::vector<size_type> delimiter_map{};
    delimiter_map.reserve(sources.size());
    auto d_buffer     = rmm::device_uvector<char>(total_source_size, stream);
    size_t bytes_read = 0;
    std::vector<std::unique_ptr<datasource::buffer>> h_buffers;
    for (auto const& source : sources) {
      if (!source->is_empty()) {
        auto data_size   = (range_size != 0) ? range_size : source->size();
        auto destination = reinterpret_cast<uint8_t*>(d_buffer.data()) + bytes_read;
        if (source->is_device_read_preferred(data_size)) {
          bytes_read += source->device_read(range_offset, data_size, destination, stream);
        } else {
          h_buffers.emplace_back(source->host_read(range_offset, data_size));
          auto const& h_buffer = h_buffers.back();
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            destination, h_buffer->data(), h_buffer->size(), cudaMemcpyDefault, stream.value()));
          bytes_read += h_buffer->size();
        }
        delimiter_map.push_back(bytes_read);
        bytes_read += num_delimiter_chars;
      }
    }

    // If this is a multi-file source, we scatter the JSON line delimiters between files
    if (sources.size() > 1) {
      static_assert(num_delimiter_chars == 1,
                    "Currently only single-character delimiters are supported");
      auto const delimiter_source = thrust::make_constant_iterator('\n');
      auto const d_delimiter_map  = cudf::detail::make_device_uvector_async(
        host_span<size_type const>{delimiter_map.data(), delimiter_map.size() - 1},
        stream,
        rmm::mr::get_current_device_resource());
      thrust::scatter(rmm::exec_policy_nosync(stream),
                      delimiter_source,
                      delimiter_source + d_delimiter_map.size(),
                      d_delimiter_map.data(),
                      d_buffer.data());
    }

    stream.synchronize();
    return d_buffer;

  } else {
    auto buffer = std::vector<uint8_t>(total_source_size);
    // Single read because only a single compressed source is supported
    // Reading to host because decompression of a single block is much faster on the CPU
    sources[0]->host_read(range_offset, total_source_size, buffer.data());
    auto const uncomp_data = decompress(compression, buffer);
    return cudf::detail::make_device_uvector_sync(
      host_span<char const>{reinterpret_cast<char const*>(uncomp_data.data()), uncomp_data.size()},
      stream,
      rmm::mr::get_current_device_resource());
  }
}

size_type find_first_delimiter_in_chunk(host_span<std::unique_ptr<cudf::io::datasource>> sources,
                                        json_reader_options const& reader_opts,
                                        char const delimiter,
                                        rmm::cuda_stream_view stream)
{
  auto const buffer = ingest_raw_input(sources,
                                       reader_opts.get_compression(),
                                       reader_opts.get_byte_range_offset(),
                                       reader_opts.get_byte_range_size(),
                                       stream);
  return find_first_delimiter(buffer, delimiter, stream);
}

bool should_load_whole_source(json_reader_options const& opts, size_t source_size)
{
  auto const range_offset = opts.get_byte_range_offset();
  auto const range_size   = opts.get_byte_range_size();
  return range_offset == 0 and (range_size == 0 or range_size >= source_size);
}

/**
 * @brief Get the byte range between record starts and ends starting from the given range.
 *
 * if get_byte_range_offset == 0, then we can skip the first delimiter search
 * if get_byte_range_offset != 0, then we need to search for the first delimiter in given range.
 * if not found, skip this chunk, if found, then search for first delimiter in next range until we
 * find a delimiter. Use this as actual range for parsing.
 *
 * @param sources Data sources to read from
 * @param reader_opts JSON reader options with range offset and range size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Byte range for parsing
 */
auto get_record_range_raw_input(host_span<std::unique_ptr<datasource>> sources,
                                json_reader_options const& reader_opts,
                                rmm::cuda_stream_view stream)
{
  size_t const total_source_size = sources_size(sources, 0, 0);
  rmm::device_uvector<char> merged(0, stream);

  rmm::device_uvector<char> cur_chunk_buf = ingest_raw_input(sources,
                                                             reader_opts.get_compression(),
                                                             reader_opts.get_byte_range_offset(),
                                                             reader_opts.get_byte_range_size(),
                                                             stream);
  auto first_delim_pos                    = reader_opts.get_byte_range_offset() == 0
                                              ? 0
                                              : find_first_delimiter(cur_chunk_buf, '\n', stream);
  if (first_delim_pos == -1) {
    return rmm::device_uvector<char>{0, stream};
  } else if (!should_load_whole_source(reader_opts, sources[0]->size()) &&
             cur_chunk_buf.back_element(stream) != '\n' &&
             reader_opts.get_byte_range_offset() + reader_opts.get_byte_range_size() <
               total_source_size) {
    // Find next delimiter
    /*
     * NOTE: heuristic for choosing subchunk size: geometric mean of minimum subchunk size (set to
     * 10kb) and the byte range size
     */
    std::int64_t next_delim_pos = -1;
    constexpr int num_subchunks = 10;  // per byte_range_size
    auto geometric_mean         = [](double a, double b) { return std::pow(a * b, 0.5); };
    size_t size_per_subchunk =
      geometric_mean(reader_opts.get_byte_range_size() / num_subchunks, 10000);
    size_t next_subchunk_start =
      reader_opts.get_byte_range_offset() + reader_opts.get_byte_range_size();
    std::vector<rmm::device_uvector<char>> subchunk_buffers;

    while (next_subchunk_start < total_source_size && next_delim_pos == -1) {
      subchunk_buffers.emplace_back(ingest_raw_input(
        sources, reader_opts.get_compression(), next_subchunk_start, size_per_subchunk, stream));
      next_delim_pos = find_first_delimiter(subchunk_buffers.back(), '\n', stream);
      if (next_delim_pos == -1) { next_subchunk_start += size_per_subchunk; }
    }
    if (next_delim_pos == -1)
      next_delim_pos = total_source_size - (next_subchunk_start - size_per_subchunk);

    merged.resize(
      cur_chunk_buf.size() + ((subchunk_buffers.size() - 1) * size_per_subchunk) + next_delim_pos,
      stream);
    size_t offset = cur_chunk_buf.size() - first_delim_pos;
    if (subchunk_buffers.size() >= 3) {
      std::vector<rmm::cuda_stream_view> copy_streams =
        cudf::detail::fork_streams(stream, subchunk_buffers.size() - 1);
      for (size_t i = 0; i < subchunk_buffers.size() - 1; i++) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(merged.data() + offset,
                                      subchunk_buffers[i].data(),
                                      size_per_subchunk,
                                      cudaMemcpyDeviceToDevice,
                                      copy_streams[i]));
        offset += size_per_subchunk;
      }
      cudf::detail::join_streams(copy_streams, stream);
    } else if (subchunk_buffers.size() == 2) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(merged.data() + offset,
                                    subchunk_buffers[0].data(),
                                    size_per_subchunk,
                                    cudaMemcpyDeviceToDevice,
                                    stream));
      offset += size_per_subchunk;
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(merged.data() + offset,
                                  subchunk_buffers.back().data(),
                                  next_delim_pos,
                                  cudaMemcpyDeviceToDevice,
                                  stream));
  } else {
    merged.resize(cur_chunk_buf.size() - first_delim_pos, stream);
  }

  CUDF_CUDA_TRY(cudaMemcpyAsync(merged.data(),
                                cur_chunk_buf.data() + first_delim_pos,
                                cur_chunk_buf.size() - first_delim_pos,
                                cudaMemcpyDeviceToDevice,
                                stream));
  stream.synchronize();
  return merged;
}

table_with_metadata read_json(host_span<std::unique_ptr<datasource>> sources,
                              json_reader_options const& reader_opts,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  if (reader_opts.is_enabled_legacy()) {
    return legacy::read_json(sources, reader_opts, stream, mr);
  }

  if (reader_opts.get_byte_range_offset() != 0 or reader_opts.get_byte_range_size() != 0) {
    CUDF_EXPECTS(reader_opts.is_enabled_lines(),
                 "Specifying a byte range is supported only for JSON Lines");
    CUDF_EXPECTS(sources.size() == 1,
                 "Specifying a byte range is supported only for a single source");
  }

  if (sources.size() > 1) {
    CUDF_EXPECTS(reader_opts.get_compression() == compression_type::NONE,
                 "Multiple compressed inputs are not supported");
    CUDF_EXPECTS(reader_opts.is_enabled_lines(),
                 "Multiple inputs are supported only for JSON Lines format");
  }

  auto buffer = get_record_range_raw_input(sources, reader_opts, stream);

  // If input JSON buffer has single quotes and option to normalize single quotes is enabled,
  // invoke pre-processing FST
  if (reader_opts.is_enabled_normalize_single_quotes()) {
    buffer =
      normalize_single_quotes(std::move(buffer), stream, rmm::mr::get_current_device_resource());
  }

  // If input JSON buffer has unquoted spaces and tabs and option to normalize whitespaces is
  // enabled, invoke pre-processing FST
  if (reader_opts.is_enabled_normalize_whitespace()) {
    buffer =
      normalize_whitespace(std::move(buffer), stream, rmm::mr::get_current_device_resource());
  }

  return device_parse_nested_json(buffer, reader_opts, stream, mr);
  // For debug purposes, use host_parse_nested_json()
}

}  // namespace cudf::io::json::detail
