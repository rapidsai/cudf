/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "compact_protocol_reader.hpp"
#include "io/parquet/parquet.hpp"
#include "reader_impl_helpers.hpp"

#include <cudf/detail/utilities/logger.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <future>
#include <numeric>
#include <optional>

namespace cudf::io::parquet::detail {

namespace {
/**
 * @brief Asynchronously reads bloom filters to device.
 *
 * @param sources Dataset sources
 * @param num_chunks Number of total column chunks to read
 * @param bloom_filter_data Devicebuffers to hold bloom filter bitsets for each chunk
 * @param bloom_filter_offsets Bloom filter offsets for all chunks
 * @param bloom_filter_sizes Bloom filter sizes for all chunks
 * @param chunk_source_map Association between each column chunk and its source
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return A future object for reading synchronization
 */
std::future<void> read_bloom_filters_async(
  host_span<std::unique_ptr<datasource> const> sources,
  size_t num_chunks,
  cudf::host_span<rmm::device_buffer> bloom_filter_data,
  cudf::host_span<std::optional<int64_t>> bloom_filter_offsets,
  cudf::host_span<std::optional<int32_t>> bloom_filter_sizes,
  std::vector<size_type> const& chunk_source_map,
  rmm::cuda_stream_view stream)
{
  // Read tasks for bloom filter data
  std::vector<std::future<size_t>> read_tasks;

  // Read bloom filters for all column chunks
  std::for_each(
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(num_chunks),
    [&](auto const chunk) {
      // Read bloom filter if present
      if (bloom_filter_offsets[chunk].has_value()) {
        auto const bloom_filter_offset = bloom_filter_offsets[chunk].value();
        // If Bloom filter size (header + bitset) is available, just read the entire thing.
        // Else just read 256 bytes which will contain the entire header and may contain the
        // entire bitset as well.
        auto constexpr bloom_filter_header_size_guess = 256;
        auto const initial_read_size =
          bloom_filter_sizes[chunk].value_or(bloom_filter_header_size_guess);

        // Read an initial buffer from source
        auto& source = sources[chunk_source_map[chunk]];
        auto buffer  = source->host_read(bloom_filter_offset, initial_read_size);

        // Deserialize the Bloom filter header from the buffer.
        BloomFilterHeader header;
        CompactProtocolReader cp{buffer->data(), buffer->size()};
        cp.read(&header);

        // Test if header is valid.
        auto const is_header_valid =
          (header.num_bytes % 32) == 0 and
          header.compression.compression == BloomFilterCompression::Compression::UNCOMPRESSED and
          header.algorithm.algorithm == BloomFilterAlgorithm::Algorithm::SPLIT_BLOCK and
          header.hash.hash == BloomFilterHash::Hash::XXHASH;

        // Do not read if the bloom filter is invalid
        if (not is_header_valid) {
          CUDF_LOG_WARN("Encountered an invalid bloom filter header. Skipping");
          return;
        }

        // Bloom filter header size
        auto const bloom_filter_header_size = static_cast<int64_t>(cp.bytecount());
        auto const bitset_size              = header.num_bytes;

        // Check if we already read in the filter bitset in the initial read.
        if (initial_read_size >= bloom_filter_header_size + bitset_size) {
          bloom_filter_data[chunk] =
            rmm::device_buffer(buffer->data() + bloom_filter_header_size, bitset_size, stream);
        }
        // Read the bitset from datasource.
        else {
          auto const bitset_offset = bloom_filter_offset + bloom_filter_header_size;
          // Directly read to device if preferred
          if (source->is_device_read_preferred(bitset_size)) {
            bloom_filter_data[chunk] = rmm::device_buffer(bitset_size, stream);
            auto future_read_size =
              source->device_read_async(bitset_offset,
                                        bitset_size,
                                        static_cast<uint8_t*>(bloom_filter_data[chunk].data()),
                                        stream);

            read_tasks.emplace_back(std::move(future_read_size));
          } else {
            buffer                   = source->host_read(bitset_offset, bitset_size);
            bloom_filter_data[chunk] = rmm::device_buffer(buffer->data(), buffer->size(), stream);
          }
        }
      }
    });
  auto sync_fn = [](decltype(read_tasks) read_tasks) {
    for (auto& task : read_tasks) {
      task.wait();
    }
  };
  return std::async(std::launch::deferred, sync_fn, std::move(read_tasks));
}

}  // namespace

std::vector<rmm::device_buffer> aggregate_reader_metadata::read_bloom_filters(
  host_span<std::unique_ptr<datasource> const> sources,
  host_span<std::vector<size_type> const> row_group_indices,
  host_span<int const> column_schemas,
  rmm::cuda_stream_view stream) const
{
  // Number of total row groups to process.
  auto const num_row_groups = std::accumulate(
    row_group_indices.begin(),
    row_group_indices.end(),
    size_t{0},
    [](size_t sum, auto const& per_file_row_groups) { return sum + per_file_row_groups.size(); });

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = column_schemas.size();
  auto const num_chunks        = num_row_groups * num_input_columns;

  // Association between each column chunk and its source
  std::vector<size_type> chunk_source_map(num_chunks);

  // Keep track of column chunk file offsets
  std::vector<std::optional<int64_t>> bloom_filter_offsets(num_chunks);
  std::vector<std::optional<int32_t>> bloom_filter_sizes(num_chunks);

  // Gather all bloom filter offsets and sizes.
  size_type chunk_count = 0;

  // For all data sources
  std::for_each(thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator(row_group_indices.size()),
                [&](auto const src_index) {
                  // Get all row group indices in the data source
                  auto const& rg_indices = row_group_indices[src_index];
                  // For all row groups
                  std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto const rg_index) {
                    // For all column chunks
                    std::for_each(
                      column_schemas.begin(), column_schemas.end(), [&](auto const schema_idx) {
                        auto& col_meta = get_column_metadata(rg_index, src_index, schema_idx);

                        // Get bloom filter offsets and sizes
                        bloom_filter_offsets[chunk_count] = col_meta.bloom_filter_offset;
                        bloom_filter_sizes[chunk_count]   = col_meta.bloom_filter_length;

                        // Map each column chunk to its source index
                        chunk_source_map[chunk_count] = src_index;
                        chunk_count++;
                      });
                  });
                });

  // Do we have any bloom filters
  if (std::any_of(bloom_filter_offsets.cbegin(),
                  bloom_filter_offsets.cend(),
                  [](auto const offset) { return offset.has_value(); })) {
    // Create a vector to store bloom filter data
    std::vector<rmm::device_buffer> bloom_filter_data(num_chunks);

    // Wait on bloom filter read tasks
    read_bloom_filters_async(sources,
                             num_chunks,
                             bloom_filter_data,
                             bloom_filter_offsets,
                             bloom_filter_sizes,
                             chunk_source_map,
                             stream)
      .wait();
    // Return the vector
    return bloom_filter_data;
  }
  return {};
}

}  // namespace cudf::io::parquet::detail
