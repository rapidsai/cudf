/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "reader_impl.hpp"

#include <io/comp/nvcomp_adapter.hpp>
#include <io/utilities/config_utils.hpp>
#include <io/utilities/time_utils.cuh>

#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/logical.h>

#include <numeric>

namespace cudf::io::detail::parquet {

namespace {

/**
 * @brief Generate depth remappings for repetition and definition levels.
 *
 * When dealing with columns that contain lists, we must examine incoming
 * repetition and definition level pairs to determine what range of output nesting
 * is indicated when adding new values.  This function generates the mappings of
 * the R/D levels to those start/end bounds
 *
 * @param remap Maps column schema index to the R/D remapping vectors for that column
 * @param src_col_schema The column schema to generate the new mapping for
 * @param md File metadata information
 */
void generate_depth_remappings(std::map<int, std::pair<std::vector<int>, std::vector<int>>>& remap,
                               int src_col_schema,
                               aggregate_reader_metadata const& md)
{
  // already generated for this level
  if (remap.find(src_col_schema) != remap.end()) { return; }
  auto schema   = md.get_schema(src_col_schema);
  int max_depth = md.get_output_nesting_depth(src_col_schema);

  CUDF_EXPECTS(remap.find(src_col_schema) == remap.end(),
               "Attempting to remap a schema more than once");
  auto inserted =
    remap.insert(std::pair<int, std::pair<std::vector<int>, std::vector<int>>>{src_col_schema, {}});
  auto& depth_remap = inserted.first->second;

  std::vector<int>& rep_depth_remap = (depth_remap.first);
  rep_depth_remap.resize(schema.max_repetition_level + 1);
  std::vector<int>& def_depth_remap = (depth_remap.second);
  def_depth_remap.resize(schema.max_definition_level + 1);

  // the key:
  // for incoming level values  R/D
  // add values starting at the shallowest nesting level X has repetition level R
  // until you reach the deepest nesting level Y that corresponds to the repetition level R1
  // held by the nesting level that has definition level D
  //
  // Example: a 3 level struct with a list at the bottom
  //
  //                     R / D   Depth
  // level0              0 / 1     0
  //   level1            0 / 2     1
  //     level2          0 / 3     2
  //       list          0 / 3     3
  //         element     1 / 4     4
  //
  // incoming R/D : 0, 0  -> add values from depth 0 to 3   (def level 0 always maps to depth 0)
  // incoming R/D : 0, 1  -> add values from depth 0 to 3
  // incoming R/D : 0, 2  -> add values from depth 0 to 3
  // incoming R/D : 1, 4  -> add values from depth 4 to 4
  //
  // Note : the -validity- of values is simply checked by comparing the incoming D value against the
  // D value of the given nesting level (incoming D >= the D for the nesting level == valid,
  // otherwise NULL).  The tricky part is determining what nesting levels to add values at.
  //
  // For schemas with no repetition level (no lists), X is always 0 and Y is always max nesting
  // depth.
  //

  // compute "X" from above
  for (int s_idx = schema.max_repetition_level; s_idx >= 0; s_idx--) {
    auto find_shallowest = [&](int r) {
      int shallowest = -1;
      int cur_depth  = max_depth - 1;
      int schema_idx = src_col_schema;
      while (schema_idx > 0) {
        auto cur_schema = md.get_schema(schema_idx);
        if (cur_schema.max_repetition_level == r) {
          // if this is a repeated field, map it one level deeper
          shallowest = cur_schema.is_stub() ? cur_depth + 1 : cur_depth;
        }
        // if it's one-level encoding list
        else if (cur_schema.is_one_level_list()) {
          shallowest = cur_depth - 1;
        }
        if (!cur_schema.is_stub()) { cur_depth--; }
        schema_idx = cur_schema.parent_idx;
      }
      return shallowest;
    };
    rep_depth_remap[s_idx] = find_shallowest(s_idx);
  }

  // compute "Y" from above
  for (int s_idx = schema.max_definition_level; s_idx >= 0; s_idx--) {
    auto find_deepest = [&](int d) {
      SchemaElement prev_schema;
      int schema_idx = src_col_schema;
      int r1         = 0;
      while (schema_idx > 0) {
        SchemaElement cur_schema = md.get_schema(schema_idx);
        if (cur_schema.max_definition_level == d) {
          // if this is a repeated field, map it one level deeper
          r1 = cur_schema.is_stub() ? prev_schema.max_repetition_level
                                    : cur_schema.max_repetition_level;
          break;
        }
        prev_schema = cur_schema;
        schema_idx  = cur_schema.parent_idx;
      }

      // we now know R1 from above. return the deepest nesting level that has the
      // same repetition level
      schema_idx = src_col_schema;
      int depth  = max_depth - 1;
      while (schema_idx > 0) {
        SchemaElement cur_schema = md.get_schema(schema_idx);
        if (cur_schema.max_repetition_level == r1) {
          // if this is a repeated field, map it one level deeper
          depth = cur_schema.is_stub() ? depth + 1 : depth;
          break;
        }
        if (!cur_schema.is_stub()) { depth--; }
        prev_schema = cur_schema;
        schema_idx  = cur_schema.parent_idx;
      }
      return depth;
    };
    def_depth_remap[s_idx] = find_deepest(s_idx);
  }
}

/**
 * @brief Function that returns the required the number of bits to store a value
 */
template <typename T = uint8_t>
[[nodiscard]] T required_bits(uint32_t max_level)
{
  return static_cast<T>(CompactProtocolReader::NumRequiredBits(max_level));
}

/**
 * @brief Converts cuDF units to Parquet units.
 *
 * @return A tuple of Parquet type width, Parquet clock rate and Parquet decimal type.
 */
[[nodiscard]] std::tuple<int32_t, int32_t, int8_t> conversion_info(type_id column_type_id,
                                                                   type_id timestamp_type_id,
                                                                   parquet::Type physical,
                                                                   int8_t converted,
                                                                   int32_t length)
{
  int32_t type_width = (physical == parquet::FIXED_LEN_BYTE_ARRAY) ? length : 0;
  int32_t clock_rate = 0;
  if (column_type_id == type_id::INT8 or column_type_id == type_id::UINT8) {
    type_width = 1;  // I32 -> I8
  } else if (column_type_id == type_id::INT16 or column_type_id == type_id::UINT16) {
    type_width = 2;  // I32 -> I16
  } else if (column_type_id == type_id::INT32) {
    type_width = 4;  // str -> hash32
  } else if (is_chrono(data_type{column_type_id})) {
    clock_rate = to_clockrate(timestamp_type_id);
  }

  int8_t converted_type = converted;
  if (converted_type == parquet::DECIMAL && column_type_id != type_id::FLOAT64 &&
      not cudf::is_fixed_point(data_type{column_type_id})) {
    converted_type = parquet::UNKNOWN;  // Not converting to float64 or decimal
  }
  return std::make_tuple(type_width, clock_rate, converted_type);
}

/**
 * @brief Reads compressed page data to device memory
 *
 * @param sources Dataset sources
 * @param page_data Buffers to hold compressed page data for each chunk
 * @param chunks List of column chunk descriptors
 * @param begin_chunk Index of first column chunk to read
 * @param end_chunk Index after the last column chunk to read
 * @param column_chunk_offsets File offset for all chunks
 * @param chunk_source_map Association between each column chunk and its source
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return A future object for reading synchronization
 */
[[nodiscard]] std::future<void> read_column_chunks_async(
  std::vector<std::unique_ptr<datasource>> const& sources,
  std::vector<std::unique_ptr<datasource::buffer>>& page_data,
  hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
  size_t begin_chunk,
  size_t end_chunk,
  const std::vector<size_t>& column_chunk_offsets,
  std::vector<size_type> const& chunk_source_map,
  rmm::cuda_stream_view stream)
{
  // Transfer chunk data, coalescing adjacent chunks
  std::vector<std::future<size_t>> read_tasks;
  for (size_t chunk = begin_chunk; chunk < end_chunk;) {
    const size_t io_offset   = column_chunk_offsets[chunk];
    size_t io_size           = chunks[chunk].compressed_size;
    size_t next_chunk        = chunk + 1;
    const bool is_compressed = (chunks[chunk].codec != parquet::Compression::UNCOMPRESSED);
    while (next_chunk < end_chunk) {
      const size_t next_offset = column_chunk_offsets[next_chunk];
      const bool is_next_compressed =
        (chunks[next_chunk].codec != parquet::Compression::UNCOMPRESSED);
      if (next_offset != io_offset + io_size || is_next_compressed != is_compressed) {
        // Can't merge if not contiguous or mixing compressed and uncompressed
        // Not coalescing uncompressed with compressed chunks is so that compressed buffers can be
        // freed earlier (immediately after decompression stage) to limit peak memory requirements
        break;
      }
      io_size += chunks[next_chunk].compressed_size;
      next_chunk++;
    }
    if (io_size != 0) {
      auto& source = sources[chunk_source_map[chunk]];
      if (source->is_device_read_preferred(io_size)) {
        auto buffer        = rmm::device_buffer(io_size, stream);
        auto fut_read_size = source->device_read_async(
          io_offset, io_size, static_cast<uint8_t*>(buffer.data()), stream);
        read_tasks.emplace_back(std::move(fut_read_size));
        page_data[chunk] = datasource::buffer::create(std::move(buffer));
      } else {
        auto const buffer = source->host_read(io_offset, io_size);
        page_data[chunk] =
          datasource::buffer::create(rmm::device_buffer(buffer->data(), buffer->size(), stream));
      }
      auto d_compdata = page_data[chunk]->data();
      do {
        chunks[chunk].compressed_data = d_compdata;
        d_compdata += chunks[chunk].compressed_size;
      } while (++chunk != next_chunk);
    } else {
      chunk = next_chunk;
    }
  }
  auto sync_fn = [](decltype(read_tasks) read_tasks) {
    for (auto& task : read_tasks) {
      task.wait();
    }
  };
  return std::async(std::launch::deferred, sync_fn, std::move(read_tasks));
}

/**
 * @brief Return the number of total pages from the given column chunks.
 *
 * @param chunks List of column chunk descriptors
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return The total number of pages
 */
[[nodiscard]] size_t count_page_headers(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                        rmm::cuda_stream_view stream)
{
  size_t total_pages = 0;

  chunks.host_to_device(stream);
  gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), stream);
  chunks.device_to_host(stream, true);

  for (size_t c = 0; c < chunks.size(); c++) {
    total_pages += chunks[c].num_data_pages + chunks[c].num_dict_pages;
  }

  return total_pages;
}

/**
 * @brief Decode the page information from the given column chunks.
 *
 * @param chunks List of column chunk descriptors
 * @param pages List of page information
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void decode_page_headers(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                         hostdevice_vector<gpu::PageInfo>& pages,
                         rmm::cuda_stream_view stream)
{
  // IMPORTANT : if you change how pages are stored within a chunk (dist pages, then data pages),
  // please update preprocess_nested_columns to reflect this.
  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    chunks[c].max_num_pages = chunks[c].num_data_pages + chunks[c].num_dict_pages;
    chunks[c].page_info     = pages.device_ptr(page_count);
    page_count += chunks[c].max_num_pages;
  }

  chunks.host_to_device(stream);
  gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), stream);
  pages.device_to_host(stream, true);
}

/**
 * @brief Decompresses the page data, at page granularity.
 *
 * @param chunks List of column chunk descriptors
 * @param pages List of page information
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Device buffer to decompressed page data
 */
[[nodiscard]] rmm::device_buffer decompress_page_data(
  hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
  hostdevice_vector<gpu::PageInfo>& pages,
  rmm::cuda_stream_view stream)
{
  auto for_each_codec_page = [&](parquet::Compression codec, const std::function<void(size_t)>& f) {
    for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
      const auto page_stride = chunks[c].max_num_pages;
      if (chunks[c].codec == codec) {
        for (int k = 0; k < page_stride; k++) {
          f(page_count + k);
        }
      }
      page_count += page_stride;
    }
  };

  // Brotli scratch memory for decompressing
  rmm::device_buffer debrotli_scratch;

  // Count the exact number of compressed pages
  size_t num_comp_pages    = 0;
  size_t total_decomp_size = 0;

  struct codec_stats {
    parquet::Compression compression_type = UNCOMPRESSED;
    size_t num_pages                      = 0;
    int32_t max_decompressed_size         = 0;
    size_t total_decomp_size              = 0;
  };

  std::array codecs{codec_stats{parquet::GZIP},
                    codec_stats{parquet::SNAPPY},
                    codec_stats{parquet::BROTLI},
                    codec_stats{parquet::ZSTD}};

  auto is_codec_supported = [&codecs](int8_t codec) {
    if (codec == parquet::UNCOMPRESSED) return true;
    return std::find_if(codecs.begin(), codecs.end(), [codec](auto& cstats) {
             return codec == cstats.compression_type;
           }) != codecs.end();
  };
  CUDF_EXPECTS(std::all_of(chunks.begin(),
                           chunks.end(),
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
    if (codec.compression_type == parquet::BROTLI && codec.num_pages > 0) {
      debrotli_scratch.resize(get_gpu_debrotli_scratch_size(codec.num_pages), stream);
    }
  }

  // Dispatch batches of pages to decompress for each codec
  rmm::device_buffer decomp_pages(total_decomp_size, stream);

  std::vector<device_span<uint8_t const>> comp_in;
  comp_in.reserve(num_comp_pages);
  std::vector<device_span<uint8_t>> comp_out;
  comp_out.reserve(num_comp_pages);

  // vectors to save v2 def and rep level data, if any
  std::vector<device_span<uint8_t const>> copy_in;
  copy_in.reserve(num_comp_pages);
  std::vector<device_span<uint8_t>> copy_out;
  copy_out.reserve(num_comp_pages);

  rmm::device_uvector<compression_result> comp_res(num_comp_pages, stream);
  thrust::fill(rmm::exec_policy(stream),
               comp_res.begin(),
               comp_res.end(),
               compression_result{0, compression_status::FAILURE});

  size_t decomp_offset = 0;
  int32_t start_pos    = 0;
  for (const auto& codec : codecs) {
    if (codec.num_pages == 0) { continue; }

    for_each_codec_page(codec.compression_type, [&](size_t page_idx) {
      auto const dst_base = static_cast<uint8_t*>(decomp_pages.data()) + decomp_offset;
      auto& page          = pages[page_idx];
      // offset will only be non-zero for V2 pages
      auto const offset = page.def_lvl_bytes + page.rep_lvl_bytes;
      // for V2 need to copy def and rep level info into place, and then offset the
      // input and output buffers. otherwise we'd have to keep both the compressed
      // and decompressed data.
      if (offset != 0) {
        copy_in.emplace_back(page.page_data, offset);
        copy_out.emplace_back(dst_base, offset);
      }
      comp_in.emplace_back(page.page_data + offset,
                           static_cast<size_t>(page.compressed_page_size - offset));
      comp_out.emplace_back(dst_base + offset,
                            static_cast<size_t>(page.uncompressed_page_size - offset));
      page.page_data = dst_base;
      decomp_offset += page.uncompressed_page_size;
    });

    host_span<device_span<uint8_t const> const> comp_in_view{comp_in.data() + start_pos,
                                                             codec.num_pages};
    auto const d_comp_in = cudf::detail::make_device_uvector_async(comp_in_view, stream);
    host_span<device_span<uint8_t> const> comp_out_view(comp_out.data() + start_pos,
                                                        codec.num_pages);
    auto const d_comp_out = cudf::detail::make_device_uvector_async(comp_out_view, stream);
    device_span<compression_result> d_comp_res_view(comp_res.data() + start_pos, codec.num_pages);

    switch (codec.compression_type) {
      case parquet::GZIP:
        gpuinflate(d_comp_in, d_comp_out, d_comp_res_view, gzip_header_included::YES, stream);
        break;
      case parquet::SNAPPY:
        if (nvcomp_integration::is_stable_enabled()) {
          nvcomp::batched_decompress(nvcomp::compression_type::SNAPPY,
                                     d_comp_in,
                                     d_comp_out,
                                     d_comp_res_view,
                                     codec.max_decompressed_size,
                                     codec.total_decomp_size,
                                     stream);
        } else {
          gpu_unsnap(d_comp_in, d_comp_out, d_comp_res_view, stream);
        }
        break;
      case parquet::ZSTD:
        nvcomp::batched_decompress(nvcomp::compression_type::ZSTD,
                                   d_comp_in,
                                   d_comp_out,
                                   d_comp_res_view,
                                   codec.max_decompressed_size,
                                   codec.total_decomp_size,
                                   stream);
        break;
      case parquet::BROTLI:
        gpu_debrotli(d_comp_in,
                     d_comp_out,
                     d_comp_res_view,
                     debrotli_scratch.data(),
                     debrotli_scratch.size(),
                     stream);
        break;
      default: CUDF_FAIL("Unexpected decompression dispatch"); break;
    }
    start_pos += codec.num_pages;
  }

  CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(stream),
                              comp_res.begin(),
                              comp_res.end(),
                              [] __device__(auto const& res) {
                                return res.status == compression_status::SUCCESS;
                              }),
               "Error during decompression");

  // now copy the uncompressed V2 def and rep level data
  if (not copy_in.empty()) {
    auto const d_copy_in  = cudf::detail::make_device_uvector_async(copy_in, stream);
    auto const d_copy_out = cudf::detail::make_device_uvector_async(copy_out, stream);

    gpu_copy_uncompressed_blocks(d_copy_in, d_copy_out, stream);
    stream.synchronize();
  }

  // Update the page information in device memory with the updated value of
  // page_data; it now points to the uncompressed data buffer
  pages.host_to_device(stream);

  return decomp_pages;
}

}  // namespace

void reader::impl::allocate_nesting_info()
{
  auto const& chunks      = _file_itm_data.chunks;
  auto& pages             = _file_itm_data.pages_info;
  auto& page_nesting_info = _file_itm_data.page_nesting_info;

  // compute total # of page_nesting infos needed and allocate space. doing this in one
  // buffer to keep it to a single gpu allocation
  size_t const total_page_nesting_infos = std::accumulate(
    chunks.host_ptr(), chunks.host_ptr() + chunks.size(), 0, [&](int total, auto& chunk) {
      // the schema of the input column
      auto const& schema                    = _metadata->get_schema(chunk.src_col_schema);
      auto const per_page_nesting_info_size = max(
        schema.max_definition_level + 1, _metadata->get_output_nesting_depth(chunk.src_col_schema));
      return total + (per_page_nesting_info_size * chunk.num_data_pages);
    });

  page_nesting_info = hostdevice_vector<gpu::PageNestingInfo>{total_page_nesting_infos, _stream};

  // retrieve from the gpu so we can update
  pages.device_to_host(_stream, true);

  // update pointers in the PageInfos
  int target_page_index = 0;
  int src_info_index    = 0;
  for (size_t idx = 0; idx < chunks.size(); idx++) {
    int src_col_schema                    = chunks[idx].src_col_schema;
    auto& schema                          = _metadata->get_schema(src_col_schema);
    auto const per_page_nesting_info_size = std::max(
      schema.max_definition_level + 1, _metadata->get_output_nesting_depth(src_col_schema));

    // skip my dict pages
    target_page_index += chunks[idx].num_dict_pages;
    for (int p_idx = 0; p_idx < chunks[idx].num_data_pages; p_idx++) {
      pages[target_page_index + p_idx].nesting = page_nesting_info.device_ptr() + src_info_index;
      pages[target_page_index + p_idx].num_nesting_levels = per_page_nesting_info_size;

      src_info_index += per_page_nesting_info_size;
    }
    target_page_index += chunks[idx].num_data_pages;
  }

  // copy back to the gpu
  pages.host_to_device(_stream);

  // fill in
  int nesting_info_index = 0;
  std::map<int, std::pair<std::vector<int>, std::vector<int>>> depth_remapping;
  for (size_t idx = 0; idx < chunks.size(); idx++) {
    int src_col_schema = chunks[idx].src_col_schema;

    // schema of the input column
    auto& schema = _metadata->get_schema(src_col_schema);
    // real depth of the output cudf column hierarchy (1 == no nesting, 2 == 1 level, etc)
    int max_depth = _metadata->get_output_nesting_depth(src_col_schema);

    // # of nesting infos stored per page for this column
    auto const per_page_nesting_info_size = std::max(schema.max_definition_level + 1, max_depth);

    // if this column has lists, generate depth remapping
    std::map<int, std::pair<std::vector<int>, std::vector<int>>> depth_remapping;
    if (schema.max_repetition_level > 0) {
      generate_depth_remappings(depth_remapping, src_col_schema, *_metadata);
    }

    // fill in host-side nesting info
    int schema_idx  = src_col_schema;
    auto cur_schema = _metadata->get_schema(schema_idx);
    int cur_depth   = max_depth - 1;
    while (schema_idx > 0) {
      // stub columns (basically the inner field of a list scheme element) are not real columns.
      // we can ignore them for the purposes of output nesting info
      if (!cur_schema.is_stub()) {
        // initialize each page within the chunk
        for (int p_idx = 0; p_idx < chunks[idx].num_data_pages; p_idx++) {
          gpu::PageNestingInfo* pni =
            &page_nesting_info[nesting_info_index + (p_idx * per_page_nesting_info_size)];

          // if we have lists, set our start and end depth remappings
          if (schema.max_repetition_level > 0) {
            auto remap = depth_remapping.find(src_col_schema);
            CUDF_EXPECTS(remap != depth_remapping.end(),
                         "Could not find depth remapping for schema");
            std::vector<int> const& rep_depth_remap = (remap->second.first);
            std::vector<int> const& def_depth_remap = (remap->second.second);

            for (size_t m = 0; m < rep_depth_remap.size(); m++) {
              pni[m].start_depth = rep_depth_remap[m];
            }
            for (size_t m = 0; m < def_depth_remap.size(); m++) {
              pni[m].end_depth = def_depth_remap[m];
            }
          }

          // values indexed by output column index
          pni[cur_depth].max_def_level = cur_schema.max_definition_level;
          pni[cur_depth].max_rep_level = cur_schema.max_repetition_level;
          pni[cur_depth].size          = 0;
        }

        // move up the hierarchy
        cur_depth--;
      }

      // next schema
      schema_idx = cur_schema.parent_idx;
      cur_schema = _metadata->get_schema(schema_idx);
    }

    nesting_info_index += (per_page_nesting_info_size * chunks[idx].num_data_pages);
  }

  // copy nesting info to the device
  page_nesting_info.host_to_device(_stream);
}

void reader::impl::load_and_decompress_data(std::vector<row_group_info> const& row_groups_info,
                                            size_type num_rows)
{
  // This function should never be called if `num_rows == 0`.
  CUDF_EXPECTS(num_rows > 0, "Number of reading rows must not be zero.");

  auto& raw_page_data    = _file_itm_data.raw_page_data;
  auto& decomp_page_data = _file_itm_data.decomp_page_data;
  auto& chunks           = _file_itm_data.chunks;
  auto& pages_info       = _file_itm_data.pages_info;

  // Descriptors for all the chunks that make up the selected columns
  const auto num_input_columns = _input_columns.size();
  const auto num_chunks        = row_groups_info.size() * num_input_columns;
  chunks                       = hostdevice_vector<gpu::ColumnChunkDesc>(0, num_chunks, _stream);

  // Association between each column chunk and its source
  std::vector<size_type> chunk_source_map(num_chunks);

  // Tracker for eventually deallocating compressed and uncompressed data
  raw_page_data = std::vector<std::unique_ptr<datasource::buffer>>(num_chunks);

  // Keep track of column chunk file offsets
  std::vector<size_t> column_chunk_offsets(num_chunks);

  // Initialize column chunk information
  size_t total_decompressed_size = 0;
  auto remaining_rows            = num_rows;
  std::vector<std::future<void>> read_rowgroup_tasks;
  for (const auto& rg : row_groups_info) {
    const auto& row_group       = _metadata->get_row_group(rg.index, rg.source_index);
    auto const row_group_start  = rg.start_row;
    auto const row_group_source = rg.source_index;
    auto const row_group_rows   = std::min<int>(remaining_rows, row_group.num_rows);
    auto const io_chunk_idx     = chunks.size();

    // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
    for (size_t i = 0; i < num_input_columns; ++i) {
      auto col = _input_columns[i];
      // look up metadata
      auto& col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);
      auto& schema   = _metadata->get_schema(col.schema_idx);

      auto [type_width, clock_rate, converted_type] =
        conversion_info(to_type_id(schema, _strings_to_categorical, _timestamp_type.id()),
                        _timestamp_type.id(),
                        schema.type,
                        schema.converted_type,
                        schema.type_length);

      column_chunk_offsets[chunks.size()] =
        (col_meta.dictionary_page_offset != 0)
          ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
          : col_meta.data_page_offset;

      chunks.push_back(gpu::ColumnChunkDesc(col_meta.total_compressed_size,
                                            nullptr,
                                            col_meta.num_values,
                                            schema.type,
                                            type_width,
                                            row_group_start,
                                            row_group_rows,
                                            schema.max_definition_level,
                                            schema.max_repetition_level,
                                            _metadata->get_output_nesting_depth(col.schema_idx),
                                            required_bits(schema.max_definition_level),
                                            required_bits(schema.max_repetition_level),
                                            col_meta.codec,
                                            converted_type,
                                            schema.logical_type,
                                            schema.decimal_scale,
                                            clock_rate,
                                            i,
                                            col.schema_idx));

      // Map each column chunk to its column index and its source index
      chunk_source_map[chunks.size() - 1] = row_group_source;

      if (col_meta.codec != Compression::UNCOMPRESSED) {
        total_decompressed_size += col_meta.total_uncompressed_size;
      }
    }
    // Read compressed chunk data to device memory
    read_rowgroup_tasks.push_back(read_column_chunks_async(_sources,
                                                           raw_page_data,
                                                           chunks,
                                                           io_chunk_idx,
                                                           chunks.size(),
                                                           column_chunk_offsets,
                                                           chunk_source_map,
                                                           _stream));

    remaining_rows -= row_group.num_rows;
  }
  for (auto& task : read_rowgroup_tasks) {
    task.wait();
  }
  CUDF_EXPECTS(remaining_rows <= 0, "All rows data must be read.");

  // Process dataset chunk pages into output columns
  auto const total_pages = count_page_headers(chunks, _stream);
  pages_info             = hostdevice_vector<gpu::PageInfo>(total_pages, total_pages, _stream);

  if (total_pages > 0) {
    // decoding of column/page information
    decode_page_headers(chunks, pages_info, _stream);
    if (total_decompressed_size > 0) {
      decomp_page_data = decompress_page_data(chunks, pages_info, _stream);
      // Free compressed data
      for (size_t c = 0; c < chunks.size(); c++) {
        if (chunks[c].codec != parquet::Compression::UNCOMPRESSED) {
          raw_page_data[c].reset();
          // TODO: Check if this is called
        }
      }
    }

    // build output column info
    // walk the schema, building out_buffers that mirror what our final cudf columns will look
    // like. important : there is not necessarily a 1:1 mapping between input columns and output
    // columns. For example, parquet does not explicitly store a ColumnChunkDesc for struct
    // columns. The "structiness" is simply implied by the schema.  For example, this schema:
    //  required group field_id=1 name {
    //    required binary field_id=2 firstname (String);
    //    required binary field_id=3 middlename (String);
    //    required binary field_id=4 lastname (String);
    // }
    // will only contain 3 columns of data (firstname, middlename, lastname).  But of course
    // "name" is a struct column that we want to return, so we have to make sure that we
    // create it ourselves.
    // std::vector<output_column_info> output_info = build_output_column_info();

    // nesting information (sizes, etc) stored -per page-
    // note : even for flat schemas, we allocate 1 level of "nesting" info
    allocate_nesting_info();
  }
}

void reader::impl::allocate_columns(size_t min_row, size_t total_rows, bool uses_custom_row_bounds)
{
  auto const& chunks = _file_itm_data.chunks;
  auto& pages        = _file_itm_data.pages_info;

  // iterate over all input columns and allocate any associated output
  // buffers if they are not part of a list hierarchy. mark down
  // if we have any list columns that need further processing.
  bool has_lists = false;
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    auto const& input_col  = _input_columns[idx];
    size_t const max_depth = input_col.nesting_depth();

    auto* cols = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // if this has a list parent, we will have to do further work in gpu::PreprocessColumnData
      // to know how big this buffer actually is.
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) {
        has_lists = true;
      }
      // if we haven't already processed this column because it is part of a struct hierarchy
      else if (out_buf.size == 0) {
        // add 1 for the offset if this is a list column
        out_buf.create(
          out_buf.type.id() == type_id::LIST && l_idx < max_depth ? total_rows + 1 : total_rows,
          _stream,
          _mr);
      }
    }
  }

  // if we have columns containing lists, further preprocessing is necessary.
  if (has_lists) {
    gpu::PreprocessColumnData(pages,
                              chunks,
                              _input_columns,
                              _output_buffers,
                              total_rows,
                              min_row,
                              uses_custom_row_bounds,
                              _stream,
                              _mr);
    _stream.synchronize();
  }
}

}  // namespace cudf::io::detail::parquet
