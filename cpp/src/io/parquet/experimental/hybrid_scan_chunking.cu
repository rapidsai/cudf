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
#include "io/comp/comp.hpp"
#include "io/comp/gpuinflate.hpp"
#include "io/comp/nvcomp_adapter.hpp"
#include "io/parquet/compact_protocol_reader.hpp"
#include "io/parquet/reader_impl_chunking.hpp"
#include "io/utilities/time_utils.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include <numeric>

namespace cudf::experimental::io::parquet::detail {

namespace {

namespace nvcomp = cudf::io::detail::nvcomp;
using cudf::io::detail::compression_result;
using cudf::io::detail::compression_status;

struct cumulative_page_info {
  size_t end_row_index;  // end row index (start_row + num_rows for the corresponding page)
  size_t size_bytes;     // cumulative size in bytes
  int key;               // schema index
};

/**
 * @brief Functor which returns the compressed data size for a chunk
 */
struct get_chunk_compressed_size {
  __device__ size_t operator()(cudf::io::parquet::detail::ColumnChunkDesc const& chunk) const
  {
    return chunk.compressed_size;
  }
};

/**
 * @brief Converts cuDF units to Parquet units.
 *
 * @return A tuple of Parquet clock rate and Parquet decimal type.
 */
[[nodiscard]] std::tuple<int32_t, std::optional<cudf::io::parquet::detail::LogicalType>>
conversion_info(type_id column_type_id,
                type_id timestamp_type_id,
                cudf::io::parquet::detail::Type physical,
                std::optional<cudf::io::parquet::detail::LogicalType> logical_type)
{
  int32_t const clock_rate =
    is_chrono(data_type{column_type_id}) ? cudf::io::to_clockrate(timestamp_type_id) : 0;

  // TODO(ets): this is leftover from the original code, but will we ever output decimal as
  // anything but fixed point?
  if (logical_type.has_value() and
      logical_type->type == cudf::io::parquet::detail::LogicalType::DECIMAL) {
    // if decimal but not outputting as float or decimal, then convert to no logical type
    if (column_type_id != type_id::FLOAT64 and
        not cudf::is_fixed_point(data_type{column_type_id})) {
      return {clock_rate, std::nullopt};
    }
  }

  return {clock_rate, std::move(logical_type)};
}

/**
 * @brief Return the required number of bits to store a value.
 */
template <typename T = uint8_t>
[[nodiscard]] T required_bits(uint32_t max_level)
{
  return static_cast<T>(
    cudf::io::parquet::detail::CompactProtocolReader::NumRequiredBits(max_level));
}

struct page_span {
  size_t start, end;
};

/**
 * @brief Return the span of page indices for a given column index

 */
struct get_page_span_by_column {
  cudf::device_span<size_type const> page_offsets;

  __device__ page_span operator()(size_t i) const
  {
    return {static_cast<size_t>(page_offsets[i]), static_cast<size_t>(page_offsets[i + 1])};
  }
};

/**
 * @brief Return the size of a span
 *
 */
struct get_span_size {
  CUDF_HOST_DEVICE size_t operator()(page_span const& s) const { return s.end - s.start; }
};

/**
 * @brief Decompresses a set of pages contained in the set of chunks.
 *
 * This function handles the case where `pages` is only a subset of all available
 * pages in `chunks`.
 *
 * @param chunks List of column chunk descriptors
 * @param pages List of page information
 * @param dict_pages If true, decompress dictionary pages only. Otherwise decompress non-dictionary
 * pages only.
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Device buffer to decompressed page data
 */
[[nodiscard]] rmm::device_buffer decompress_page_data(
  cudf::detail::hostdevice_span<cudf::io::parquet::detail::ColumnChunkDesc const> chunks,
  cudf::detail::hostdevice_span<cudf::io::parquet::detail::PageInfo> pages,
  bool dict_pages,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  auto for_each_codec_page = [&](cudf::io::parquet::detail::Compression codec,
                                 std::function<void(size_t)> const& f) {
    for (size_t p = 0; p < pages.size(); p++) {
      if (chunks[pages[p].chunk_idx].codec == codec &&
          ((dict_pages &&
            (pages[p].flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY)) ||
           (!dict_pages &&
            !(pages[p].flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY)))) {
        f(p);
      }
    }
  };

  // Brotli scratch memory for decompressing
  rmm::device_buffer debrotli_scratch;

  // Count the exact number of compressed pages
  size_t num_comp_pages    = 0;
  size_t total_decomp_size = 0;

  struct codec_stats {
    cudf::io::parquet::detail::Compression compression_type =
      cudf::io::parquet::detail::UNCOMPRESSED;
    size_t num_pages              = 0;
    int32_t max_decompressed_size = 0;
    size_t total_decomp_size      = 0;
  };

  std::array codecs{codec_stats{cudf::io::parquet::detail::GZIP},
                    codec_stats{cudf::io::parquet::detail::SNAPPY},
                    codec_stats{cudf::io::parquet::detail::BROTLI},
                    codec_stats{cudf::io::parquet::detail::ZSTD},
                    codec_stats{cudf::io::parquet::detail::LZ4_RAW}};

  auto is_codec_supported = [&codecs](int8_t codec) {
    if (codec == cudf::io::parquet::detail::UNCOMPRESSED) return true;
    return std::find_if(codecs.begin(), codecs.end(), [codec](auto& cstats) {
             return codec == cstats.compression_type;
           }) != codecs.end();
  };
  CUDF_EXPECTS(std::all_of(chunks.host_begin(),
                           chunks.host_end(),
                           [&is_codec_supported](auto const& chunk) {
                             return is_codec_supported(chunk.codec);
                           }),
               "Unsupported compression type");

  for (auto& codec : codecs) {
    for_each_codec_page(codec.compression_type, [&](size_t page) {
      auto page_uncomp_size = pages[page].uncompressed_page_size;
      total_decomp_size += page_uncomp_size;
      codec.total_decomp_size += page_uncomp_size;
      codec.max_decompressed_size = std::max(codec.max_decompressed_size, page_uncomp_size);
      codec.num_pages++;
      num_comp_pages++;
    });
    if (codec.compression_type == cudf::io::parquet::detail::BROTLI && codec.num_pages > 0) {
      debrotli_scratch.resize(cudf::io::detail::get_gpu_debrotli_scratch_size(codec.num_pages),
                              stream);
    }
  }

  // Dispatch batches of pages to decompress for each codec.
  // Buffer needs to be padded, required by `gpuDecodePageData`.
  rmm::device_buffer decomp_pages(
    cudf::util::round_up_safe(total_decomp_size, cudf::io::detail::BUFFER_PADDING_MULTIPLE),
    stream);

  auto comp_in =
    cudf::detail::make_empty_host_vector<device_span<uint8_t const>>(num_comp_pages, stream);
  auto comp_out =
    cudf::detail::make_empty_host_vector<device_span<uint8_t>>(num_comp_pages, stream);

  // vectors to save v2 def and rep level data, if any
  auto copy_in =
    cudf::detail::make_empty_host_vector<device_span<uint8_t const>>(num_comp_pages, stream);
  auto copy_out =
    cudf::detail::make_empty_host_vector<device_span<uint8_t>>(num_comp_pages, stream);

  rmm::device_uvector<compression_result> comp_res(num_comp_pages, stream);
  thrust::fill(rmm::exec_policy_nosync(stream),
               comp_res.begin(),
               comp_res.end(),
               compression_result{0, compression_status::FAILURE});

  size_t decomp_offset = 0;
  for (auto const& codec : codecs) {
    if (codec.num_pages == 0) { continue; }

    for_each_codec_page(codec.compression_type, [&](size_t page_idx) {
      auto const dst_base = static_cast<uint8_t*>(decomp_pages.data()) + decomp_offset;
      auto& page          = pages[page_idx];
      // offset will only be non-zero for V2 pages
      auto const offset = page.lvl_bytes[cudf::io::parquet::detail::level_type::DEFINITION] +
                          page.lvl_bytes[cudf::io::parquet::detail::level_type::REPETITION];
      // for V2 need to copy def and rep level info into place, and then offset the
      // input and output buffers. otherwise we'd have to keep both the compressed
      // and decompressed data.
      if (offset != 0) {
        copy_in.push_back({page.page_data, static_cast<size_t>(offset)});
        copy_out.push_back({dst_base, static_cast<size_t>(offset)});
      }
      comp_in.push_back(
        {page.page_data + offset, static_cast<size_t>(page.compressed_page_size - offset)});
      comp_out.push_back(
        {dst_base + offset, static_cast<size_t>(page.uncompressed_page_size - offset)});
      page.page_data = dst_base;
      decomp_offset += page.uncompressed_page_size;
    });
  }
  auto d_comp_in = cudf::detail::make_device_uvector_async(
    comp_in, stream, cudf::get_current_device_resource_ref());
  auto d_comp_out = cudf::detail::make_device_uvector_async(
    comp_out, stream, cudf::get_current_device_resource_ref());

  int32_t start_pos = 0;
  for (auto const& codec : codecs) {
    if (codec.num_pages == 0) { continue; }

    device_span<device_span<uint8_t const> const> d_comp_in_view{d_comp_in.data() + start_pos,
                                                                 codec.num_pages};

    device_span<device_span<uint8_t> const> d_comp_out_view(d_comp_out.data() + start_pos,
                                                            codec.num_pages);

    device_span<compression_result> d_comp_res_view(comp_res.data() + start_pos, codec.num_pages);

    switch (codec.compression_type) {
      case cudf::io::parquet::detail::GZIP:
        if (cudf::io::nvcomp_integration::is_all_enabled()) {
          nvcomp::batched_decompress(nvcomp::compression_type::GZIP,
                                     d_comp_in_view,
                                     d_comp_out_view,
                                     d_comp_res_view,
                                     codec.max_decompressed_size,
                                     codec.total_decomp_size,
                                     stream);
        } else {
          gpuinflate(d_comp_in_view,
                     d_comp_out_view,
                     d_comp_res_view,
                     cudf::io::detail::gzip_header_included::YES,
                     stream);
        }
        break;
      case cudf::io::parquet::detail::SNAPPY:
        if (cudf::io::nvcomp_integration::is_stable_enabled()) {
          nvcomp::batched_decompress(nvcomp::compression_type::SNAPPY,
                                     d_comp_in_view,
                                     d_comp_out_view,
                                     d_comp_res_view,
                                     codec.max_decompressed_size,
                                     codec.total_decomp_size,
                                     stream);
        } else {
          gpu_unsnap(d_comp_in_view, d_comp_out, d_comp_res_view, stream);
        }
        break;
      case cudf::io::parquet::detail::ZSTD:
        nvcomp::batched_decompress(nvcomp::compression_type::ZSTD,
                                   d_comp_in_view,
                                   d_comp_out_view,
                                   d_comp_res_view,
                                   codec.max_decompressed_size,
                                   codec.total_decomp_size,
                                   stream);
        break;
      case cudf::io::parquet::detail::BROTLI:
        gpu_debrotli(d_comp_in_view,
                     d_comp_out_view,
                     d_comp_res_view,
                     debrotli_scratch.data(),
                     debrotli_scratch.size(),
                     stream);
        break;
      case cudf::io::parquet::detail::LZ4_RAW:
        nvcomp::batched_decompress(nvcomp::compression_type::LZ4,
                                   d_comp_in_view,
                                   d_comp_out_view,
                                   d_comp_res_view,
                                   codec.max_decompressed_size,
                                   codec.total_decomp_size,
                                   stream);
        break;
      default: CUDF_FAIL("Unexpected decompression dispatch"); break;
    }
    start_pos += codec.num_pages;
  }

  CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(stream),
                              comp_res.begin(),
                              comp_res.end(),
                              cuda::proclaim_return_type<bool>([] __device__(auto const& res) {
                                return res.status == compression_status::SUCCESS;
                              })),
               "Error during decompression");

  // now copy the uncompressed V2 def and rep level data
  if (not copy_in.empty()) {
    auto const d_copy_in = cudf::detail::make_device_uvector_async(
      copy_in, stream, cudf::get_current_device_resource_ref());
    auto const d_copy_out = cudf::detail::make_device_uvector_async(
      copy_out, stream, cudf::get_current_device_resource_ref());

    cudf::io::detail::gpu_copy_uncompressed_blocks(d_copy_in, d_copy_out, stream);
    stream.synchronize();
  }

  pages.host_to_device_async(stream);

  stream.synchronize();
  return decomp_pages;
}

struct flat_column_num_rows {
  cudf::io::parquet::detail::ColumnChunkDesc const* chunks;

  __device__ size_type operator()(cudf::io::parquet::detail::PageInfo const& page) const
  {
    // ignore dictionary pages and pages belonging to any column containing repetition (lists)
    if ((page.flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY) ||
        (chunks[page.chunk_idx].max_level[cudf::io::parquet::detail::level_type::REPETITION] > 0)) {
      return 0;
    }
    return page.num_rows;
  }
};

struct row_counts_nonzero {
  __device__ bool operator()(size_type count) const { return count > 0; }
};

struct row_counts_different {
  size_type const expected;
  __device__ bool operator()(size_type count) const { return (count != 0) && (count != expected); }
};

/**
 * @brief Detect malformed parquet input data.
 *
 * We have seen cases where parquet files can be oddly malformed. This function specifically
 * detects one case in particular:
 *
 * - When you have a file containing N rows
 * - For some reason, the sum total of the number of rows over all pages for a given column
 *   is != N
 *
 * @param pages All pages to be decoded
 * @param chunks Chunk data
 * @param expected_row_count Expected row count, if applicable
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void detect_malformed_pages(device_span<cudf::io::parquet::detail::PageInfo const> pages,
                            device_span<cudf::io::parquet::detail::ColumnChunkDesc const> chunks,
                            std::optional<size_t> expected_row_count,
                            rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // sum row counts for all non-dictionary, non-list columns. other columns will be indicated as 0
  rmm::device_uvector<size_type> row_counts(pages.size(),
                                            stream);  // worst case:  num keys == num pages
  auto const size_iter =
    thrust::make_transform_iterator(pages.begin(), flat_column_num_rows{chunks.data()});
  auto const row_counts_begin = row_counts.begin();
  auto page_keys              = make_page_key_iterator(pages);
  auto const row_counts_end   = thrust::reduce_by_key(rmm::exec_policy(stream),
                                                    page_keys,
                                                    page_keys + pages.size(),
                                                    size_iter,
                                                    thrust::make_discard_iterator(),
                                                    row_counts_begin)
                                .second;

  // make sure all non-zero row counts are the same
  rmm::device_uvector<size_type> compacted_row_counts(pages.size(), stream);
  auto const compacted_row_counts_begin = compacted_row_counts.begin();
  auto const compacted_row_counts_end   = thrust::copy_if(rmm::exec_policy(stream),
                                                        row_counts_begin,
                                                        row_counts_end,
                                                        compacted_row_counts_begin,
                                                        row_counts_nonzero{});
  if (compacted_row_counts_end != compacted_row_counts_begin) {
    auto const found_row_count = static_cast<size_t>(compacted_row_counts.element(0, stream));

    // if we somehow don't match the expected row count from the row groups themselves
    if (expected_row_count.has_value()) {
      CUDF_EXPECTS(expected_row_count.value() == found_row_count,
                   "Encountered malformed parquet page data (unexpected row count in page data)");
    }

    // all non-zero row counts must be the same
    auto const chk =
      thrust::count_if(rmm::exec_policy(stream),
                       compacted_row_counts_begin,
                       compacted_row_counts_end,
                       row_counts_different{static_cast<size_type>(found_row_count)});
    CUDF_EXPECTS(chk == 0,
                 "Encountered malformed parquet page data (row count mismatch in page data)");
  }
}

struct decompression_info {
  cudf::io::parquet::detail::Compression codec;
  size_t num_pages;
  size_t max_page_decompressed_size;
  size_t total_decompressed_size;
};

/**
 * @brief Functor which retrieves per-page decompression information.
 *
 */
struct get_decomp_info {
  device_span<cudf::io::parquet::detail::ColumnChunkDesc const> chunks;

  __device__ decompression_info operator()(cudf::io::parquet::detail::PageInfo const& p) const
  {
    return {static_cast<cudf::io::parquet::detail::Compression>(chunks[p.chunk_idx].codec),
            1,
            static_cast<size_t>(p.uncompressed_page_size),
            static_cast<size_t>(p.uncompressed_page_size)};
  }
};

/**
 * @brief Functor which accumulates per-page decompression information.
 *
 */
struct decomp_sum {
  __device__ decompression_info operator()(decompression_info const& a,
                                           decompression_info const& b) const
  {
    return {a.codec,
            a.num_pages + b.num_pages,
            cuda::std::max(a.max_page_decompressed_size, b.max_page_decompressed_size),
            a.total_decompressed_size + b.total_decompressed_size};
  }
};

/**
 * @brief Functor which returns total scratch space required based on computed decompression_info
 * data.
 *
 */
struct get_decomp_scratch {
  size_t operator()(decompression_info const& di) const
  {
    switch (di.codec) {
      case cudf::io::parquet::detail::UNCOMPRESSED:
      case cudf::io::parquet::detail::GZIP: return 0;

      case cudf::io::parquet::detail::BROTLI:
        return cudf::io::detail::get_gpu_debrotli_scratch_size(di.num_pages);

      case cudf::io::parquet::detail::SNAPPY:
        if (cudf::io::nvcomp_integration::is_stable_enabled()) {
          return nvcomp::batched_decompress_temp_size(nvcomp::compression_type::SNAPPY,
                                                      di.num_pages,
                                                      di.max_page_decompressed_size,
                                                      di.total_decompressed_size);
        } else {
          return 0;
        }
        break;

      case cudf::io::parquet::detail::ZSTD:
        return nvcomp::batched_decompress_temp_size(nvcomp::compression_type::ZSTD,
                                                    di.num_pages,
                                                    di.max_page_decompressed_size,
                                                    di.total_decompressed_size);
      case cudf::io::parquet::detail::LZ4_RAW:
        return nvcomp::batched_decompress_temp_size(nvcomp::compression_type::LZ4,
                                                    di.num_pages,
                                                    di.max_page_decompressed_size,
                                                    di.total_decompressed_size);

      default: CUDF_FAIL("Invalid compression codec for parquet decompression");
    }
  }
};

/**
 * @brief Add the cost of decompression codec scratch space to the per-page cumulative
 * size information.
 *
 */
[[maybe_unused]] void include_decompression_scratch_size(
  device_span<cudf::io::parquet::detail::ColumnChunkDesc const> chunks,
  device_span<cudf::io::parquet::detail::PageInfo const> pages,
  device_span<cumulative_page_info> c_info,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() == c_info.size(),
               "Encountered page/cumulative_page_info size mismatch");

  auto page_keys = make_page_key_iterator(pages);

  // per-codec page counts and decompression sizes
  rmm::device_uvector<decompression_info> decomp_info(pages.size(), stream);
  auto decomp_iter = thrust::make_transform_iterator(pages.begin(), get_decomp_info{chunks});
  thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(stream),
                                page_keys,
                                page_keys + pages.size(),
                                decomp_iter,
                                decomp_info.begin(),
                                cuda::std::equal_to<int32_t>{},
                                decomp_sum{});

  // retrieve to host so we can call nvcomp to get compression scratch sizes
  auto h_decomp_info = cudf::detail::make_host_vector_sync(decomp_info, stream);
  auto temp_cost     = cudf::detail::make_host_vector<size_t>(pages.size(), stream);
  thrust::transform(thrust::host,
                    h_decomp_info.begin(),
                    h_decomp_info.end(),
                    temp_cost.begin(),
                    get_decomp_scratch{});

  // add to the cumulative_page_info data
  rmm::device_uvector<size_t> d_temp_cost = cudf::detail::make_device_uvector_async(
    temp_cost, stream, cudf::get_current_device_resource_ref());
  auto iter = thrust::make_counting_iterator(size_t{0});
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   iter,
                   iter + pages.size(),
                   [temp_cost = d_temp_cost.begin(), c_info = c_info.begin()] __device__(size_t i) {
                     c_info[i].size_bytes += temp_cost[i];
                   });
  stream.synchronize();
}

}  // anonymous namespace

void impl::create_global_chunk_info(cudf::io::parquet_reader_options const& options)
{
  auto const num_rows         = _file_itm_data.global_num_rows;
  auto const& row_groups_info = _file_itm_data.row_groups;
  auto& chunks                = _file_itm_data.chunks;

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_chunks        = row_groups_info.size() * num_input_columns;

  // Mapping of input column to page index column
  std::vector<size_type> column_mapping;

  if (_has_page_index and not row_groups_info.empty()) {
    // use first row group to define mappings (assumes same schema for each file)
    auto const& rg      = row_groups_info[0];
    auto const& columns = _metadata->get_row_group(rg.index, rg.source_index).columns;
    column_mapping.resize(num_input_columns);
    std::transform(
      _input_columns.begin(), _input_columns.end(), column_mapping.begin(), [&](auto const& col) {
        // translate schema_idx into something we can use for the page indexes
        if (auto it = std::find_if(columns.begin(),
                                   columns.end(),
                                   [&](auto const& col_chunk) {
                                     return col_chunk.schema_idx ==
                                            _metadata->map_schema_index(col.schema_idx,
                                                                        rg.source_index);
                                   });
            it != columns.end()) {
          return std::distance(columns.begin(), it);
        }
        CUDF_FAIL("cannot find column mapping");
      });
  }

  // Initialize column chunk information
  auto remaining_rows = num_rows;
  auto skip_rows      = _file_itm_data.global_skip_rows;
  for (auto const& rg : row_groups_info) {
    auto const& row_group      = _metadata->get_row_group(rg.index, rg.source_index);
    auto const row_group_start = rg.start_row;
    auto const row_group_rows  = std::min<int>(remaining_rows, row_group.num_rows);

    // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
    for (size_t i = 0; i < num_input_columns; ++i) {
      auto col = _input_columns[i];
      // look up metadata
      auto& col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);
      auto& schema   = _metadata->get_schema(
        _metadata->map_schema_index(col.schema_idx, rg.source_index), rg.source_index);

      auto [clock_rate, logical_type] =
        conversion_info(to_type_id(schema,
                                   options.is_enabled_convert_strings_to_categories(),
                                   options.get_timestamp_type().id()),
                        options.get_timestamp_type().id(),
                        schema.type,
                        schema.logical_type);

      // for lists, estimate the number of bytes per row. this is used by the subpass reader to
      // determine where to split the decompression boundaries
      float const list_bytes_per_row_est =
        schema.max_repetition_level > 0 && row_group.num_rows > 0
          ? static_cast<float>(col_meta.total_uncompressed_size) /
              static_cast<float>(row_group.num_rows)
          : 0.0f;

      // grab the column_chunk_info for each chunk (if it exists)
      cudf::io::parquet::detail::column_chunk_info const* const chunk_info =
        _has_page_index ? &rg.column_chunks.value()[column_mapping[i]] : nullptr;

      chunks.push_back(cudf::io::parquet::detail::ColumnChunkDesc(
        col_meta.total_compressed_size,
        nullptr,
        col_meta.num_values,
        schema.type,
        schema.type_length,
        row_group_start,
        row_group_rows,
        schema.max_definition_level,
        schema.max_repetition_level,
        _metadata->get_output_nesting_depth(col.schema_idx),
        required_bits(schema.max_definition_level),
        required_bits(schema.max_repetition_level),
        col_meta.codec,
        logical_type,
        clock_rate,
        i,
        col.schema_idx,
        chunk_info,
        list_bytes_per_row_est,
        schema.type == cudf::io::parquet::detail::BYTE_ARRAY and
          options.is_enabled_convert_strings_to_categories(),
        rg.source_index));
    }
    // Adjust for skip_rows when updating the remaining rows after the first group
    remaining_rows -=
      (skip_rows) ? std::min<int>(rg.start_row + row_group.num_rows - skip_rows, remaining_rows)
                  : row_group_rows;
    // Set skip_rows = 0 as it is no longer needed for subsequent row_groups
    skip_rows = 0;
  }
}

void impl::compute_input_passes()
{
  // at this point, row_groups has already been filtered down to just the row groups we need to
  // handle optional skip_rows/num_rows parameters.
  auto const& row_groups_info = _file_itm_data.row_groups;

  // read everything in a single pass.
  _file_itm_data.input_pass_row_group_offsets.push_back(0);
  _file_itm_data.input_pass_row_group_offsets.push_back(row_groups_info.size());
  _file_itm_data.input_pass_start_row_count.push_back(0);
  auto rg_row_count = cudf::detail::make_counting_transform_iterator(0, [&](size_t i) {
    auto const& rgi       = row_groups_info[i];
    auto const& row_group = _metadata->get_row_group(rgi.index, rgi.source_index);
    return row_group.num_rows;
  });
  _file_itm_data.input_pass_start_row_count.push_back(
    std::reduce(rg_row_count, rg_row_count + row_groups_info.size()));
  return;
}

void impl::compute_output_chunks_for_subpass()
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  // simple case : no chunk size, no splits
  subpass.output_chunk_read_info.push_back({subpass.skip_rows, subpass.num_rows});
  return;
}

void impl::handle_chunking(std::vector<rmm::device_buffer> column_chunk_buffers,
                           cudf::io::parquet_reader_options const& options)
{
  // if this is our first time in here, setup the first pass.
  if (!_pass_itm_data) {
    // setup the next pass
    setup_next_pass(std::move(column_chunk_buffers), options);
  }

  auto& pass = *_pass_itm_data;

  // if we already have a subpass in flight.
  if (pass.subpass != nullptr) {
    // if it still has more chunks in flight, there's nothing more to do
    if (pass.subpass->current_output_chunk < pass.subpass->output_chunk_read_info.size()) {
      return;
    }

    // increment rows processed
    pass.processed_rows += pass.subpass->num_rows;

    // release the old subpass (will free memory)
    pass.subpass.reset();

    // otherwise we are done with the pass entirely
    if (pass.processed_rows == pass.num_rows) {
      // release the old pass
      _pass_itm_data.reset();

      _file_itm_data._current_input_pass++;
      // no more passes. we are absolutely done with this file.
      if (_file_itm_data._current_input_pass == _file_itm_data.num_passes()) { return; }

      // setup the next pass
      setup_next_pass(std::move(column_chunk_buffers), options);
    }
  }

  // setup the next sub pass
  setup_next_subpass(options);
}

void impl::setup_next_pass(std::vector<rmm::device_buffer> column_chunk_buffers,
                           cudf::io::parquet_reader_options const& options)
{
  auto const num_passes = _file_itm_data.num_passes();
  CUDF_EXPECTS(num_passes == 1, "");

  // always create the pass struct, even if we end up with no work.
  // this will also cause the previous pass information to be deleted
  _pass_itm_data = std::make_unique<cudf::io::parquet::detail::pass_intermediate_data>();

  if (_file_itm_data.global_num_rows > 0 && not _file_itm_data.row_groups.empty() &&
      not _input_columns.empty() && _file_itm_data._current_input_pass < num_passes) {
    auto& pass = *_pass_itm_data;

    // setup row groups to be loaded for this pass
    auto const row_group_start =
      _file_itm_data.input_pass_row_group_offsets[_file_itm_data._current_input_pass];
    auto const row_group_end =
      _file_itm_data.input_pass_row_group_offsets[_file_itm_data._current_input_pass + 1];
    auto const num_row_groups = row_group_end - row_group_start;
    pass.row_groups.resize(num_row_groups);
    std::copy(_file_itm_data.row_groups.begin() + row_group_start,
              _file_itm_data.row_groups.begin() + row_group_end,
              pass.row_groups.begin());

    CUDF_EXPECTS(_file_itm_data._current_input_pass < num_passes,
                 "Encountered an invalid read pass index");

    auto const chunks_per_rowgroup = _input_columns.size();
    auto const num_chunks          = chunks_per_rowgroup * num_row_groups;

    auto chunk_start = _file_itm_data.chunks.begin() + (row_group_start * chunks_per_rowgroup);
    auto chunk_end   = _file_itm_data.chunks.begin() + (row_group_end * chunks_per_rowgroup);

    pass.chunks = cudf::detail::hostdevice_vector<cudf::io::parquet::detail::ColumnChunkDesc>(
      num_chunks, _stream);
    std::copy(chunk_start, chunk_end, pass.chunks.begin());

    // compute skip_rows / num_rows for this pass.
    pass.skip_rows = _file_itm_data.global_skip_rows;
    pass.num_rows  = _file_itm_data.global_num_rows;

    // load page information for the chunk. this retrieves the compressed bytes for all the
    // pages, and their headers (which we can access without decompressing)

    read_compressed_data(std::move(column_chunk_buffers));

    // detect malformed columns.
    // - we have seen some cases in the wild where we have a row group containing N
    //   rows, but the total number of rows in the pages for column X is != N. while it
    //   is possible to load this by just capping the number of rows read, we cannot tell
    //   which rows are invalid so we may be returning bad data. in addition, this mismatch
    //   confuses the chunked reader
    detect_malformed_pages(pass.pages,
                           pass.chunks,
                           (options.get_num_rows().has_value() or options.get_skip_rows() > 0)
                             ? std::nullopt
                             : std::make_optional(pass.num_rows),
                           _stream);

    // decompress dictionary data if applicable.
    if (pass.has_compressed_data) {
      pass.decomp_dict_data = decompress_page_data(pass.chunks, pass.pages, true, _stream);
    }

    // store off how much memory we've used so far. This includes the compressed page data and the
    // decompressed dictionary data. we will subtract this from the available total memory for the
    // subpasses
    auto chunk_iter =
      thrust::make_transform_iterator(pass.chunks.d_begin(), get_chunk_compressed_size{});
    pass.base_mem_size =
      pass.decomp_dict_data.size() +
      thrust::reduce(rmm::exec_policy(_stream), chunk_iter, chunk_iter + pass.chunks.size());

    // since there is only ever 1 dictionary per chunk (the first page), do it at the
    // pass level.
    build_string_dict_indices();

    _stream.synchronize();
  }
}

void impl::setup_next_subpass(cudf::io::parquet_reader_options const& options)
{
  auto& pass    = *_pass_itm_data;
  pass.subpass  = std::make_unique<cudf::io::parquet::detail::subpass_intermediate_data>();
  auto& subpass = *pass.subpass;

  auto const num_columns = _input_columns.size();

  // page_indices is an array of spans where each element N is the
  // indices into the pass.pages array that represents the subset of pages
  // for column N to use for the subpass.
  auto [page_indices, total_pages, total_expected_size] =
    [&]() -> std::tuple<rmm::device_uvector<page_span>, size_t, size_t> {
    rmm::device_uvector<page_span> page_indices(
      num_columns, _stream, cudf::get_current_device_resource_ref());
    auto iter = thrust::make_counting_iterator(0);
    thrust::transform(rmm::exec_policy_nosync(_stream),
                      iter,
                      iter + num_columns,
                      page_indices.begin(),
                      get_page_span_by_column{pass.page_offsets});
    return {std::move(page_indices), pass.pages.size(), size_t{0}};
  }();

  // check to see if we are processing the entire pass (enabling us to skip a bunch of work)
  subpass.single_subpass = total_pages == pass.pages.size();

  CUDF_EXPECTS(subpass.single_subpass, "Hybrid scan reader must read in one subpass");

  // in the single pass case, no page copying is necessary - just use what's in the pass itself
  subpass.pages = pass.pages;

  auto const h_spans = cudf::detail::make_host_vector_async(page_indices, _stream);
  subpass.pages.device_to_host_async(_stream);

  _stream.synchronize();

  subpass.column_page_count = std::vector<size_t>(num_columns);
  std::transform(
    h_spans.begin(), h_spans.end(), subpass.column_page_count.begin(), get_span_size{});

  // decompress the data for the pages in this subpass.
  if (pass.has_compressed_data) {
    subpass.decomp_page_data = decompress_page_data(pass.chunks, subpass.pages, false, _stream);
  }

  // buffers needed by the decode kernels
  {
    // nesting information (sizes, etc) stored -per page-
    // note : even for flat schemas, we allocate 1 level of "nesting" info

    allocate_nesting_info();

    // level decode space
    allocate_level_decode_space();
  }
  subpass.pages.host_to_device_async(_stream);

  // preprocess pages (computes row counts for lists, computes output chunks and computes
  // the actual row counts we will be able load out of this subpass)
  preprocess_subpass_pages(0);
}

}  // namespace cudf::experimental::io::parquet::detail
