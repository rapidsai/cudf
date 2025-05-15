
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
#include "io/comp/gpuinflate.hpp"
#include "io/comp/io_uncomp.hpp"
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
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include <numeric>

namespace cudf::io::parquet::experimental::detail {

namespace {

namespace nvcomp = cudf::io::detail::nvcomp;

using compression_result    = io::detail::compression_result;
using compression_status    = io::detail::compression_status;
using ColumnChunkDesc       = parquet::detail::ColumnChunkDesc;
using CompactProtocolReader = parquet::detail::CompactProtocolReader;
using level_type            = parquet::detail::level_type;
using PageInfo              = parquet::detail::PageInfo;

#if defined(PAGE_PRUNING_DEBUG)
void print_pages(cudf::detail::hostdevice_span<PageInfo> pages, rmm::cuda_stream_view _stream)
{
  pages.device_to_host(_stream);
  for (size_t idx = 0; idx < pages.size(); idx++) {
    auto const& p = pages[idx];
    // skip dictionary pages
    if (p.flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY) { printf("Dict"); }
    printf(
      "P(%lu, s:%d): chunk_row(%d), num_rows(%d), skipped_values(%d), skipped_leaf_values(%d), "
      "str_bytes(%d)\n",
      idx,
      p.src_col_schema,
      p.chunk_row,
      p.num_rows,
      p.skipped_values,
      p.skipped_leaf_values,
      p.str_bytes);
  }
}
#endif  // PAGE_PRUNING_DEBUG

/**
 * @brief Returns the cudf compression type and whether it is supported by the parquet writer.
 */
__host__ __device__ cuda::std::pair<compression_type, bool> parquet_compression_support(
  Compression compression)
{
  switch (compression) {
    case Compression::BROTLI: return {compression_type::BROTLI, true};
    case Compression::GZIP: return {compression_type::GZIP, true};
    case Compression::LZ4_RAW: return {compression_type::LZ4, true};
    case Compression::LZO: return {compression_type::LZO, false};
    case Compression::SNAPPY: return {compression_type::SNAPPY, true};
    case Compression::ZSTD: return {compression_type::ZSTD, true};
    case Compression::UNCOMPRESSED: return {compression_type::NONE, true};
    default: break;
  }
  return {compression_type::NONE, false};
}

/**
 * @brief Returns the string name of the Parquet compression type.
 */
[[nodiscard]] std::string parquet_compression_name(Compression compression)
{
  switch (compression) {
    case Compression::BROTLI: return "BROTLI";
    case Compression::GZIP: return "GZIP";
    case Compression::LZ4_RAW: return "LZ4_RAW";
    case Compression::LZ4: return "LZ4";
    case Compression::LZO: return "LZO";
    case Compression::SNAPPY: return "SNAPPY";
    case Compression::ZSTD: return "ZSTD";
    case Compression::UNCOMPRESSED: return "UNCOMPRESSED";
  }
  CUDF_FAIL("Unsupported Parquet compression type");
}

compression_type from_parquet_compression(Compression compression)
{
  auto const [type, supported] = parquet_compression_support(compression);
  CUDF_EXPECTS(supported, "Unsupported compression type");
  return type;
}

/**
 * @brief Stores basic information about pages compressed with a specific codec.
 */
struct codec_stats {
  Compression compression_type  = Compression::UNCOMPRESSED;
  size_t num_pages              = 0;
  int32_t max_decompressed_size = 0;
  size_t total_decomp_size      = 0;

  enum class page_selection { DICT_PAGES, NON_DICT_PAGES };

  void add_pages(host_span<ColumnChunkDesc const> chunks,
                 host_span<PageInfo> pages,
                 page_selection selection,
                 host_span<bool const> page_mask)
  {
    for (size_t page_idx = 0; page_idx < pages.size(); ++page_idx) {
      auto& page = pages[page_idx];
      if (chunks[page.chunk_idx].codec == compression_type &&
          (page.flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY) ==
            (selection == page_selection::DICT_PAGES) &&
          page_mask[page_idx]) {
        ++num_pages;
        total_decomp_size += page.uncompressed_page_size;
        max_decompressed_size = std::max(max_decompressed_size, page.uncompressed_page_size);
      }
    }
  }
};

/**
 * @brief Decompresses a mix of dictionary and non-dictionary pages from a set of column chunks.
 *
 * To avoid multiple calls to the decompression kernel, we batch pages by codec type, where the
 * batch can include both dictionary and non-dictionary pages. This allows us to decompress all
 * pages of a given codec type in one go.
 *
 * @param chunks List of column chunk descriptors
 * @param pass_pages List of page information for the pass
 * @param subpass_pages List of page information for the subpass
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return A pair of device buffers containing the decompressed data for dictionary and
 * non-dictionary pages, respectively.
 */
[[nodiscard]] std::pair<rmm::device_buffer, rmm::device_buffer> decompress_page_data(
  host_span<ColumnChunkDesc const> chunks,
  host_span<PageInfo> pass_pages,
  host_span<PageInfo> subpass_pages,
  host_span<bool const> page_mask,
  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  std::array codecs{codec_stats{Compression::BROTLI},
                    codec_stats{Compression::GZIP},
                    codec_stats{Compression::LZ4_RAW},
                    codec_stats{Compression::SNAPPY},
                    codec_stats{Compression::ZSTD}};

  auto is_codec_supported = [&codecs](Compression codec) {
    if (codec == Compression::UNCOMPRESSED) return true;
    return std::find_if(codecs.begin(), codecs.end(), [codec](auto& cstats) {
             return codec == cstats.compression_type;
           }) != codecs.end();
  };

  for (auto const& chunk : chunks) {
    CUDF_EXPECTS(is_codec_supported(chunk.codec),
                 "Unsupported Parquet compression type: " + parquet_compression_name(chunk.codec));
  }

  size_t total_pass_decomp_size = 0;
  for (auto& codec : codecs) {
    codec.add_pages(chunks, pass_pages, codec_stats::page_selection::DICT_PAGES, page_mask);
    total_pass_decomp_size += codec.total_decomp_size;
  }

  // Total number of pages to decompress, including both pass and subpass pages
  size_t num_comp_pages    = 0;
  size_t total_decomp_size = 0;
  for (auto& codec : codecs) {
    codec.add_pages(chunks, subpass_pages, codec_stats::page_selection::NON_DICT_PAGES, page_mask);
    // at this point, the codec contains info for both dictionary pass pages and data subpass pages
    total_decomp_size += codec.total_decomp_size;
    num_comp_pages += codec.num_pages;
  }

  // Dispatch batches of pages to decompress for each codec.
  // Buffer needs to be padded, required by `gpuDecodePageData`.
  rmm::device_buffer pass_decomp_pages(
    cudf::util::round_up_safe(total_pass_decomp_size, cudf::io::detail::BUFFER_PADDING_MULTIPLE),
    stream);
  auto const total_subpass_decomp_size = total_decomp_size - total_pass_decomp_size;
  rmm::device_buffer subpass_decomp_pages(
    cudf::util::round_up_safe(total_subpass_decomp_size, cudf::io::detail::BUFFER_PADDING_MULTIPLE),
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
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                             comp_res.begin(),
                             comp_res.end(),
                             compression_result{0, compression_status::FAILURE});

  auto set_parameters = [&](codec_stats const& codec,
                            host_span<PageInfo> pages,
                            void* decomp_data,
                            bool select_dict_pages,
                            size_t& decomp_offset) {
    for (size_t page_idx = 0; page_idx < pages.size(); ++page_idx) {
      auto& page = pages[page_idx];
      if (chunks[page.chunk_idx].codec == codec.compression_type &&
          (page.flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY) ==
            select_dict_pages &&
          page_mask[page_idx]) {
        auto const dst_base = static_cast<uint8_t*>(decomp_data) + decomp_offset;
        // offset will only be non-zero for V2 pages
        auto const offset =
          page.lvl_bytes[level_type::DEFINITION] + page.lvl_bytes[level_type::REPETITION];
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
      }
    }
  };

  size_t pass_decomp_offset    = 0;
  size_t subpass_decomp_offset = 0;
  for (auto const& codec : codecs) {
    if (codec.num_pages == 0) { continue; }
    set_parameters(codec, pass_pages, pass_decomp_pages.data(), true, pass_decomp_offset);
    set_parameters(codec, subpass_pages, subpass_decomp_pages.data(), false, subpass_decomp_offset);
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
    cudf::io::detail::decompress(from_parquet_compression(codec.compression_type),
                                 d_comp_in_view,
                                 d_comp_out_view,
                                 d_comp_res_view,
                                 codec.max_decompressed_size,
                                 codec.total_decomp_size,
                                 stream);

    start_pos += codec.num_pages;
  }
  // now copy the uncompressed V2 def and rep level data
  if (not copy_in.empty()) {
    auto const d_copy_in = cudf::detail::make_device_uvector_async(
      copy_in, stream, cudf::get_current_device_resource_ref());
    auto const d_copy_out = cudf::detail::make_device_uvector_async(
      copy_out, stream, cudf::get_current_device_resource_ref());

    cudf::io::detail::gpu_copy_uncompressed_blocks(d_copy_in, d_copy_out, stream);
  }

  CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(stream),
                              comp_res.begin(),
                              comp_res.end(),
                              [] __device__(auto const& res) {
                                return res.status == compression_status::SUCCESS;
                              }),
               "Error during decompression");

  return {std::move(pass_decomp_pages), std::move(subpass_decomp_pages)};
}

}  // anonymous namespace

rmm::device_buffer hybrid_scan_reader_impl::decompress_dictionary_page_data(
  cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
  cudf::detail::hostdevice_span<PageInfo> pages,
  rmm::cuda_stream_view stream)
{
  auto const page_mask = thrust::host_vector<bool>(pages.size(), true);
  return std::get<0>(decompress_page_data(chunks, pages, host_span<PageInfo>{}, page_mask, stream));
}

}  // namespace cudf::io::parquet::experimental::detail
