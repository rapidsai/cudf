/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "hybrid_scan_helpers.hpp"
#include "hybrid_scan_impl.hpp"
#include "io/parquet/parquet_gpu.hpp"
#include "io/utilities/time_utils.cuh"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/batched_memset.hpp>
#include <cudf/detail/utilities/functional.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include <bitset>
#include <limits>
#include <numeric>

namespace cudf::io::parquet::experimental::detail {

namespace {

using parquet::detail::chunk_page_info;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::PageInfo;

/**
 * @brief Converts cuDF units to Parquet units.
 *
 * @return A tuple of Parquet clock rate and Parquet decimal type.
 */
[[nodiscard]] int32_t conversion_info(type_id column_type_id,
                                      type_id timestamp_type_id,
                                      std::optional<LogicalType> logical_type)
{
  int32_t const clock_rate =
    is_chrono(data_type{column_type_id}) ? to_clockrate(timestamp_type_id) : 0;

  return clock_rate;
}

/**
 * @brief Decode the dictionary page information from each column chunk
 *
 * @param chunks Host device span of column chunk descriptors, one per input column chunk
 * @param pages Host device span of empty page headers to fill in, one per input column chunk
 * @param stream CUDA stream
 */
void decode_dictionary_page_headers(cudf::detail::hostdevice_span<ColumnChunkDesc> chunks,
                                    cudf::detail::hostdevice_span<PageInfo> pages,
                                    rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  std::vector<size_t> host_chunk_page_counts(chunks.size() + 1);
  std::transform(
    chunks.host_begin(), chunks.host_end(), host_chunk_page_counts.begin(), [](auto const& chunk) {
      return chunk.num_dict_pages;
    });
  host_chunk_page_counts[chunks.size()] = 0;

  auto chunk_page_counts = cudf::detail::make_device_uvector_async(
    host_chunk_page_counts, stream, cudf::get_current_device_resource_ref());

  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         chunk_page_counts.begin(),
                         chunk_page_counts.end(),
                         chunk_page_counts.begin(),
                         size_t{0},
                         thrust::plus<size_t>{});

  rmm::device_uvector<chunk_page_info> d_chunk_page_info(chunks.size(), stream);

  thrust::for_each(rmm::exec_policy_nosync(stream),
                   thrust::counting_iterator<cuda::std::size_t>(0),
                   thrust::counting_iterator(chunks.size()),
                   [cpi               = d_chunk_page_info.begin(),
                    chunk_page_counts = chunk_page_counts.begin(),
                    pages             = pages.device_begin()] __device__(size_t i) {
                     cpi[i].pages = &pages[chunk_page_counts[i]];
                   });

  parquet::kernel_error error_code(stream);

  DecodePageHeaders(
    chunks.device_begin(), d_chunk_page_info.begin(), chunks.size(), error_code.data(), stream);

  if (auto const error = error_code.value_sync(stream); error != 0) {
    CUDF_FAIL("Parquet header parsing failed with code(s) " +
              parquet::kernel_error::to_string(error));
  }

  // Setup dictionary page for each chunk
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   pages.device_begin(),
                   pages.device_end(),
                   [chunks = chunks.device_begin()] __device__(PageInfo const& p) {
                     if (p.flags & parquet::detail::PAGEINFO_FLAGS_DICTIONARY) {
                       chunks[p.chunk_idx].dict_page = &p;
                     }
                   });

  pages.device_to_host_async(stream);
  chunks.device_to_host_async(stream);
  stream.synchronize();
}

}  // namespace

std::tuple<bool,
           cudf::detail::hostdevice_vector<ColumnChunkDesc>,
           cudf::detail::hostdevice_vector<PageInfo>>
hybrid_scan_reader_impl::prepare_dictionaries(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<rmm::device_buffer> dictionary_page_data,
  cudf::host_span<int const> dictionary_col_schemas,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  // Create row group information for the input row group indices
  auto const row_groups_info =
    std::get<2>(_metadata->select_row_groups(row_group_indices, 0, std::nullopt));

  CUDF_EXPECTS(row_groups_info.size() * _input_columns.size() == dictionary_page_data.size(),
               "Dictionary page data size must match the number of row groups times the number of "
               "input columns");

  // Number of input columns
  auto const num_input_columns = _input_columns.size();
  // Number of column chunks
  auto const total_column_chunks = dictionary_page_data.size();

  // Boolean to check if any of the column chunnks have compressed data
  auto has_compressed_data = false;

  // Initialize column chunk descriptors
  auto chunks = cudf::detail::hostdevice_vector<cudf::io::parquet::detail::ColumnChunkDesc>(
    total_column_chunks, stream);
  auto chunk_idx = 0;

  // For all row groups
  for (auto const& rg : row_groups_info) {
    auto const& row_group = _metadata->get_row_group(rg.index, rg.source_index);

    // For all columns with dictionary page and (in)equality predicate
    for (auto col_schema_idx : dictionary_col_schemas) {
      // look up metadata
      auto& col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col_schema_idx);
      auto& schema   = _metadata->get_schema(
        _metadata->map_schema_index(col_schema_idx, rg.source_index), rg.source_index);

      // dictionary data buffer for this column chunk
      auto& dict_page_data = dictionary_page_data[chunk_idx];

      // Check if the column chunk has compressed data
      has_compressed_data =
        col_meta.codec != Compression::UNCOMPRESSED and col_meta.total_compressed_size > 0;

      auto const clock_rate = conversion_info(
        parquet::detail::to_type_id(schema,
                                    options.is_enabled_convert_strings_to_categories(),
                                    options.get_timestamp_type().id()),
        options.get_timestamp_type().id(),
        schema.logical_type);

      // Create a column chunk
      chunks[chunk_idx] = ColumnChunkDesc(static_cast<int64_t>(dict_page_data.size()),
                                          static_cast<uint8_t*>(dict_page_data.data()),
                                          col_meta.num_values,
                                          schema.type,
                                          schema.type_length,
                                          0,  // Not needed
                                          0,  // Not needed
                                          0,  // Not needed
                                          0,  // Not needed
                                          0,  // Not needed
                                          0,  // Not needed
                                          0,  // Not needed
                                          col_meta.codec,
                                          schema.logical_type,
                                          clock_rate,
                                          0,  // Not needed
                                          col_schema_idx,
                                          nullptr,  // Not needed
                                          0.0f,     // Not needed
                                          false,    // Not needed
                                          rg.source_index);
      // Set the number of dictionary and data pages
      chunks[chunk_idx].num_dict_pages = (dict_page_data.size() > 0);
      chunks[chunk_idx].num_data_pages = 0;
      chunk_idx++;
    }
  }

  // Copy the column chunk descriptors to the device
  chunks.host_to_device_async(stream);

  // Create page infos for each column chunk's dictionary page
  cudf::detail::hostdevice_vector<PageInfo> pages(total_column_chunks, stream);

  // Decode dictionary page headers
  decode_dictionary_page_headers(chunks, pages, stream);

  return {has_compressed_data, std::move(chunks), std::move(pages)};
}

}  // namespace cudf::io::parquet::experimental::detail
