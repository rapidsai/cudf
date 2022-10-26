/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include "reader_impl_helpers.cuh"

#include <io/comp/nvcomp_adapter.hpp>
#include <io/utilities/config_utils.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/logical.h>

namespace cudf::io::detail::parquet {

namespace {

void decompress_check(device_span<compression_result const> results, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(stream),
                              results.begin(),
                              results.end(),
                              [] __device__(auto const& res) {
                                return res.status == compression_status::SUCCESS;
                              }),
               "Error during decompression");
}

/**
 * @brief Recursively copy the output buffer from one to another.
 *
 * This only copies `name` and `user_data` fields, which are generated during reader construction.
 *
 * @param buff The old output buffer
 * @param new_buff The new output buffer
 */
void copy_output_buffer(column_buffer const& buff, column_buffer& new_buff)
{
  new_buff.name      = buff.name;
  new_buff.user_data = buff.user_data;
  for (auto const& child : buff.children) {
    auto& new_child = new_buff.children.emplace_back(column_buffer(child.type, child.is_nullable));
    copy_output_buffer(child, new_child);
  }
}

}  // namespace

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

std::future<void> reader::impl::read_column_chunks(
  std::vector<std::unique_ptr<datasource::buffer>>& page_data,
  hostdevice_vector<gpu::ColumnChunkDesc>& chunks,  // TODO const?
  size_t begin_chunk,
  size_t end_chunk,
  const std::vector<size_t>& column_chunk_offsets,
  std::vector<size_type> const& chunk_source_map)
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
      auto& source = _sources[chunk_source_map[chunk]];
      if (source->is_device_read_preferred(io_size)) {
        auto buffer        = rmm::device_buffer(io_size, _stream);
        auto fut_read_size = source->device_read_async(
          io_offset, io_size, static_cast<uint8_t*>(buffer.data()), _stream);
        read_tasks.emplace_back(std::move(fut_read_size));
        page_data[chunk] = datasource::buffer::create(std::move(buffer));
      } else {
        auto const buffer = source->host_read(io_offset, io_size);
        page_data[chunk] =
          datasource::buffer::create(rmm::device_buffer(buffer->data(), buffer->size(), _stream));
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

size_t reader::impl::count_page_headers(hostdevice_vector<gpu::ColumnChunkDesc>& chunks)
{
  size_t total_pages = 0;

  chunks.host_to_device(_stream);
  gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), _stream);
  chunks.device_to_host(_stream, true);

  for (size_t c = 0; c < chunks.size(); c++) {
    total_pages += chunks[c].num_data_pages + chunks[c].num_dict_pages;
  }

  return total_pages;
}

void reader::impl::decode_page_headers(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                       hostdevice_vector<gpu::PageInfo>& pages)
{
  // IMPORTANT : if you change how pages are stored within a chunk (dist pages, then data pages),
  // please update preprocess_nested_columns to reflect this.
  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    chunks[c].max_num_pages = chunks[c].num_data_pages + chunks[c].num_dict_pages;
    chunks[c].page_info     = pages.device_ptr(page_count);
    page_count += chunks[c].max_num_pages;
  }

  chunks.host_to_device(_stream);
  gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), _stream);
  pages.device_to_host(_stream, true);
}

rmm::device_buffer reader::impl::decompress_page_data(
  hostdevice_vector<gpu::ColumnChunkDesc>& chunks, hostdevice_vector<gpu::PageInfo>& pages)
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
      debrotli_scratch.resize(get_gpu_debrotli_scratch_size(codec.num_pages), _stream);
    }
  }

  // Dispatch batches of pages to decompress for each codec
  rmm::device_buffer decomp_pages(total_decomp_size, _stream);

  std::vector<device_span<uint8_t const>> comp_in;
  comp_in.reserve(num_comp_pages);
  std::vector<device_span<uint8_t>> comp_out;
  comp_out.reserve(num_comp_pages);

  // vectors to save v2 def and rep level data, if any
  std::vector<device_span<uint8_t const>> copy_in;
  copy_in.reserve(num_comp_pages);
  std::vector<device_span<uint8_t>> copy_out;
  copy_out.reserve(num_comp_pages);

  rmm::device_uvector<compression_result> comp_res(num_comp_pages, _stream);
  thrust::fill(rmm::exec_policy(_stream),
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
    auto const d_comp_in = cudf::detail::make_device_uvector_async(comp_in_view, _stream);
    host_span<device_span<uint8_t> const> comp_out_view(comp_out.data() + start_pos,
                                                        codec.num_pages);
    auto const d_comp_out = cudf::detail::make_device_uvector_async(comp_out_view, _stream);
    device_span<compression_result> d_comp_res_view(comp_res.data() + start_pos, codec.num_pages);

    switch (codec.compression_type) {
      case parquet::GZIP:
        gpuinflate(d_comp_in, d_comp_out, d_comp_res_view, gzip_header_included::YES, _stream);
        break;
      case parquet::SNAPPY:
        if (nvcomp_integration::is_stable_enabled()) {
          nvcomp::batched_decompress(nvcomp::compression_type::SNAPPY,
                                     d_comp_in,
                                     d_comp_out,
                                     d_comp_res_view,
                                     codec.max_decompressed_size,
                                     codec.total_decomp_size,
                                     _stream);
        } else {
          gpu_unsnap(d_comp_in, d_comp_out, d_comp_res_view, _stream);
        }
        break;
      case parquet::ZSTD:
        nvcomp::batched_decompress(nvcomp::compression_type::ZSTD,
                                   d_comp_in,
                                   d_comp_out,
                                   d_comp_res_view,
                                   codec.max_decompressed_size,
                                   codec.total_decomp_size,
                                   _stream);
        break;
      case parquet::BROTLI:
        gpu_debrotli(d_comp_in,
                     d_comp_out,
                     d_comp_res_view,
                     debrotli_scratch.data(),
                     debrotli_scratch.size(),
                     _stream);
        break;
      default: CUDF_FAIL("Unexpected decompression dispatch"); break;
    }
    start_pos += codec.num_pages;
  }

  decompress_check(comp_res, _stream);

  // now copy the uncompressed V2 def and rep level data
  if (not copy_in.empty()) {
    auto const d_copy_in  = cudf::detail::make_device_uvector_async(copy_in, _stream);
    auto const d_copy_out = cudf::detail::make_device_uvector_async(copy_out, _stream);

    gpu_copy_uncompressed_blocks(d_copy_in, d_copy_out, _stream);
    _stream.synchronize();
  }

  // Update the page information in device memory with the updated value of
  // page_data; it now points to the uncompressed data buffer
  pages.host_to_device(_stream);

  return decomp_pages;
}

void reader::impl::allocate_nesting_info(hostdevice_vector<gpu::ColumnChunkDesc> const& chunks,
                                         hostdevice_vector<gpu::PageInfo>& pages,
                                         hostdevice_vector<gpu::PageNestingInfo>& page_nesting_info)
{
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
    auto const type_id = to_type_id(schema, _strings_to_categorical, _timestamp_type.id());

    // skip my dict pages
    target_page_index += chunks[idx].num_dict_pages;
    for (int p_idx = 0; p_idx < chunks[idx].num_data_pages; p_idx++) {
      pages[target_page_index + p_idx].nesting = page_nesting_info.device_ptr() + src_info_index;
      pages[target_page_index + p_idx].num_nesting_levels = per_page_nesting_info_size;

      // this isn't the ideal place to be setting this value (it's not obvious this function would
      // do it) but we don't have any other places that go host->device with the pages and I'd like
      // to avoid another copy
      pages[target_page_index + p_idx].type = type_id;

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

void reader::impl::decode_page_data(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                    hostdevice_vector<gpu::PageInfo>& pages,
                                    hostdevice_vector<gpu::PageNestingInfo>& page_nesting,
                                    size_t min_row,
                                    size_t total_rows)
{
  // TODO (dm): hd_vec should have begin and end iterator members
  size_t sum_max_depths =
    std::accumulate(chunks.host_ptr(),
                    chunks.host_ptr(chunks.size()),
                    0,
                    [&](size_t cursum, gpu::ColumnChunkDesc const& chunk) {
                      return cursum + _metadata->get_output_nesting_depth(chunk.src_col_schema);
                    });

  // In order to reduce the number of allocations of hostdevice_vector, we allocate a single vector
  // to store all per-chunk pointers to nested data/nullmask. `chunk_offsets[i]` will store the
  // offset into `chunk_nested_data`/`chunk_nested_valids` for the array of pointers for chunk `i`
  auto chunk_nested_valids = hostdevice_vector<uint32_t*>(sum_max_depths, _stream);
  auto chunk_nested_data   = hostdevice_vector<void*>(sum_max_depths, _stream);
  auto chunk_offsets       = std::vector<size_t>();

  // Update chunks with pointers to column data.
  for (size_t c = 0, page_count = 0, chunk_off = 0; c < chunks.size(); c++) {
    input_column_info const& input_col = _input_columns[chunks[c].src_col_index];
    CUDF_EXPECTS(input_col.schema_idx == chunks[c].src_col_schema,
                 "Column/page schema index mismatch");

    size_t max_depth = _metadata->get_output_nesting_depth(chunks[c].src_col_schema);
    chunk_offsets.push_back(chunk_off);

    // get a slice of size `nesting depth` from `chunk_nested_valids` to store an array of pointers
    // to validity data
    auto valids              = chunk_nested_valids.host_ptr(chunk_off);
    chunks[c].valid_map_base = chunk_nested_valids.device_ptr(chunk_off);

    // get a slice of size `nesting depth` from `chunk_nested_data` to store an array of pointers to
    // out data
    auto data                  = chunk_nested_data.host_ptr(chunk_off);
    chunks[c].column_data_base = chunk_nested_data.device_ptr(chunk_off);

    chunk_off += max_depth;

    // fill in the arrays on the host.  there are some important considerations to
    // take into account here for nested columns.  specifically, with structs
    // there is sharing of output buffers between input columns.  consider this schema
    //
    //  required group field_id=1 name {
    //    required binary field_id=2 firstname (String);
    //    required binary field_id=3 middlename (String);
    //    required binary field_id=4 lastname (String);
    // }
    //
    // there are 3 input columns of data here (firstname, middlename, lastname), but
    // only 1 output column (name).  The structure of the output column buffers looks like
    // the schema itself
    //
    // struct      (name)
    //     string  (firstname)
    //     string  (middlename)
    //     string  (lastname)
    //
    // The struct column can contain validity information. the problem is, the decode
    // step for the input columns will all attempt to decode this validity information
    // because each one has it's own copy of the repetition/definition levels. but
    // since this is all happening in parallel it would mean multiple blocks would
    // be stomping all over the same memory randomly.  to work around this, we set
    // things up so that only 1 child of any given nesting level fills in the
    // data (offsets in the case of lists) or validity information for the higher
    // levels of the hierarchy that are shared.  In this case, it would mean we
    // would just choose firstname to be the one that decodes the validity for name.
    //
    // we do this by only handing out the pointers to the first child we come across.
    //
    auto* cols = &_output_columns;
    for (size_t idx = 0; idx < max_depth; idx++) {
      auto& out_buf = (*cols)[input_col.nesting[idx]];
      cols          = &out_buf.children;

      int owning_schema = out_buf.user_data & PARQUET_COLUMN_BUFFER_SCHEMA_MASK;
      if (owning_schema == 0 || owning_schema == input_col.schema_idx) {
        valids[idx] = out_buf.null_mask();
        data[idx]   = out_buf.data();
        out_buf.user_data |=
          static_cast<uint32_t>(input_col.schema_idx) & PARQUET_COLUMN_BUFFER_SCHEMA_MASK;
      } else {
        valids[idx] = nullptr;
        data[idx]   = nullptr;
      }
    }

    // column_data_base will always point to leaf data, even for nested types.
    page_count += chunks[c].max_num_pages;
  }

  chunks.host_to_device(_stream);
  chunk_nested_valids.host_to_device(_stream);
  chunk_nested_data.host_to_device(_stream);

  gpu::DecodePageData(pages, chunks, total_rows, min_row, _stream);

  _stream.synchronize();

  pages.device_to_host(_stream);
  page_nesting.device_to_host(_stream);
  _stream.synchronize();

  // for list columns, add the final offset to every offset buffer.
  // TODO : make this happen in more efficiently. Maybe use thrust::for_each
  // on each buffer.  Or potentially do it in PreprocessColumnData
  // Note : the reason we are doing this here instead of in the decode kernel is
  // that it is difficult/impossible for a given page to know that it is writing the very
  // last value that should then be followed by a terminator (because rows can span
  // page boundaries).
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    input_column_info const& input_col = _input_columns[idx];

    auto* cols = &_output_columns;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      if (out_buf.type.id() != type_id::LIST ||
          (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED)) {
        continue;
      }
      CUDF_EXPECTS(l_idx < input_col.nesting_depth() - 1, "Encountered a leaf list column");
      auto& child = (*cols)[input_col.nesting[l_idx + 1]];

      // the final offset for a list at level N is the size of it's child
      int offset = child.type.id() == type_id::LIST ? child.size - 1 : child.size;
      cudaMemcpyAsync(static_cast<int32_t*>(out_buf.data()) + (out_buf.size - 1),
                      &offset,
                      sizeof(offset),
                      cudaMemcpyHostToDevice,
                      _stream.value());
      out_buf.user_data |= PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED;
    }
  }

  // update null counts in the final column buffers
  for (size_t idx = 0; idx < pages.size(); idx++) {
    gpu::PageInfo* pi = &pages[idx];
    if (pi->flags & gpu::PAGEINFO_FLAGS_DICTIONARY) { continue; }
    gpu::ColumnChunkDesc* col          = &chunks[pi->chunk_idx];
    input_column_info const& input_col = _input_columns[col->src_col_index];

    int index                 = pi->nesting - page_nesting.device_ptr();
    gpu::PageNestingInfo* pni = &page_nesting[index];

    auto* cols = &_output_columns;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // if I wasn't the one who wrote out the validity bits, skip it
      if (chunk_nested_valids.host_ptr(chunk_offsets[pi->chunk_idx])[l_idx] == nullptr) {
        continue;
      }
      out_buf.null_count() += pni[l_idx].null_count;
    }
  }

  _stream.synchronize();
}

reader::impl::impl(std::vector<std::unique_ptr<datasource>>&& sources,
                   parquet_reader_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
  : _stream(stream), _mr(mr), _sources(std::move(sources))
{
  // Open and parse the source dataset metadata
  _metadata = std::make_unique<aggregate_reader_metadata>(_sources);

  // Override output timestamp resolution if requested
  if (options.get_timestamp_type().id() != type_id::EMPTY) {
    _timestamp_type = options.get_timestamp_type();
  }

  // Strings may be returned as either string or categorical columns
  _strings_to_categorical = options.is_enabled_convert_strings_to_categories();

  // Binary columns can be read as binary or strings
  _reader_column_schema = options.get_column_schema();

  // Select only columns required by the options
  std::tie(_input_columns, _output_columns, _output_column_schemas) =
    _metadata->select_columns(options.get_columns(),
                              options.is_enabled_use_pandas_metadata(),
                              _strings_to_categorical,
                              _timestamp_type.id());
}

reader::impl::impl(std::size_t chunk_read_limit,
                   std::vector<std::unique_ptr<datasource>>&& sources,
                   parquet_reader_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
  : impl(std::forward<std::vector<std::unique_ptr<cudf::io::datasource>>>(sources),
         options,
         stream,
         mr)
{
  _chunk_read_limit = chunk_read_limit;

  // Save the states of the output buffers for reuse in `chunk_read()`.
  for (auto const& buff : _output_columns) {
    auto& new_buff =
      _output_columns_template.emplace_back(column_buffer(buff.type, buff.is_nullable));
    copy_output_buffer(buff, new_buff);
  }
}

void reader::impl::preprocess_file_and_columns(
  size_type skip_rows,
  size_type num_rows,
  bool uses_custom_row_bounds,
  std::vector<std::vector<size_type>> const& row_group_list)
{
  if (_file_preprocessed) { return; }

  auto [skip_rows_corrected, num_rows_corrected] =
    preprocess_file(skip_rows, num_rows, row_group_list);

  if (_file_itm_data.has_data) {
    preprocess_columns(_file_itm_data.chunks,
                       _file_itm_data.pages_info,
                       skip_rows_corrected,
                       num_rows_corrected,
                       uses_custom_row_bounds,
                       _chunk_read_limit);

    if (_chunk_read_limit == 0) {  // read the whole file at once
      CUDF_EXPECTS(_chunk_read_info.size() == 1,
                   "Reading the whole file should yield only one chunk.");
    }
  }

  _file_preprocessed = true;
}

table_with_metadata reader::impl::read_chunk_internal(bool uses_custom_row_bounds)
{
  // If `_output_metadata` has been constructed, just copy it over.
  auto out_metadata = _output_metadata ? table_metadata{*_output_metadata} : table_metadata{};

  // output cudf columns as determined by the top level schema
  auto out_columns = std::vector<std::unique_ptr<column>>{};
  out_columns.reserve(_output_columns.size());

  if (!has_next()) { return finalize_output(out_metadata, out_columns); }

  auto const& read_info = _chunk_read_info[_current_read_chunk++];

  // allocate outgoing columns
  allocate_columns(_file_itm_data.chunks,
                   _file_itm_data.pages_info,
                   _chunk_itm_data,
                   read_info.skip_rows,
                   read_info.num_rows,
                   uses_custom_row_bounds);

  //  printf("read skip_rows = %d, num_rows = %d\n", (int)read_info.skip_rows,
  //  (int)read_info.num_rows);

  // decoding column data
  decode_page_data(_file_itm_data.chunks,
                   _file_itm_data.pages_info,
                   _file_itm_data.page_nesting_info,
                   read_info.skip_rows,
                   read_info.num_rows);

  // create the final output cudf columns
  for (size_t i = 0; i < _output_columns.size(); ++i) {
    auto const metadata = _reader_column_schema.has_value()
                            ? std::make_optional<reader_column_schema>((*_reader_column_schema)[i])
                            : std::nullopt;
    // Only construct `out_metadata` if `_output_metadata` has not been cached.
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info.emplace_back("");
      out_columns.emplace_back(make_column(_output_columns[i], &col_name, metadata, _stream, _mr));
    } else {
      out_columns.emplace_back(make_column(_output_columns[i], nullptr, metadata, _stream, _mr));
    }
  }

  return finalize_output(out_metadata, out_columns);
}

table_with_metadata reader::impl::finalize_output(table_metadata& out_metadata,
                                                  std::vector<std::unique_ptr<column>>& out_columns)
{
  // Create empty columns as needed (this can happen if we've ended up with no actual data to read)
  for (size_t i = out_columns.size(); i < _output_columns.size(); ++i) {
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info.emplace_back("");
      out_columns.emplace_back(io::detail::empty_like(_output_columns[i], &col_name, _stream, _mr));
    } else {
      out_columns.emplace_back(io::detail::empty_like(_output_columns[i], nullptr, _stream, _mr));
    }
  }

  if (!_output_metadata) {
    // Return column names (must match order of returned columns)
    out_metadata.column_names.resize(_output_columns.size());
    for (size_t i = 0; i < _output_column_schemas.size(); i++) {
      auto const& schema           = _metadata->get_schema(_output_column_schemas[i]);
      out_metadata.column_names[i] = schema.name;
    }

    // Return user metadata
    out_metadata.per_file_user_data = _metadata->get_key_value_metadata();
    out_metadata.user_data          = {out_metadata.per_file_user_data[0].begin(),
                              out_metadata.per_file_user_data[0].end()};

    // Finally, save the output table metadata into `_output_metadata` for reuse next time.
    _output_metadata = std::make_unique<table_metadata>(out_metadata);
  }

  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

// #define ALLOW_PLAIN_READ_CHUNK_LIMIT
table_with_metadata reader::impl::read(size_type skip_rows,
                                       size_type num_rows,
                                       bool uses_custom_row_bounds,
                                       std::vector<std::vector<size_type>> const& row_group_list)
{
#if defined(ALLOW_PLAIN_READ_CHUNK_LIMIT)
  preprocess_file_and_columns(
    skip_rows, num_rows, uses_custom_row_bounds || _chunk_read_limit > 0, row_group_list);
  return read_chunk_internal(uses_custom_row_bounds || _chunk_read_limit > 0);
#else
  CUDF_EXPECTS(_chunk_read_limit == 0, "Reading the whole file must not have non-zero byte_limit.");
  preprocess_file_and_columns(skip_rows, num_rows, uses_custom_row_bounds, row_group_list);
  return read_chunk_internal(uses_custom_row_bounds);
#endif
}

table_with_metadata reader::impl::read_chunk()
{
  // Reset the output buffers to their original states (right after reader construction).
  _output_columns.resize(0);
  for (auto const& buff : _output_columns_template) {
    auto& new_buff = _output_columns.emplace_back(column_buffer(buff.type, buff.is_nullable));
    copy_output_buffer(buff, new_buff);
  }

  preprocess_file_and_columns(0, -1, true, {});
  return read_chunk_internal(true);
}

bool reader::impl::has_next()
{
  preprocess_file_and_columns(0, -1, true, {});
  return _current_read_chunk < _chunk_read_info.size();
}

}  // namespace cudf::io::detail::parquet
