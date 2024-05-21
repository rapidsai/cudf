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

#include "error.hpp"
#include "reader_impl.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
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
#include <numeric>

namespace cudf::io::parquet::detail {
namespace {

#if defined(PREPROCESS_DEBUG)
void print_pages(cudf::detail::hostdevice_vector<PageInfo>& pages, rmm::cuda_stream_view _stream)
{
  pages.device_to_host_sync(_stream);
  for (size_t idx = 0; idx < pages.size(); idx++) {
    auto const& p = pages[idx];
    // skip dictionary pages
    if (p.flags & PAGEINFO_FLAGS_DICTIONARY) { continue; }
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
#endif  // PREPROCESS_DEBUG

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
        else if (cur_schema.is_one_level_list(md.get_schema(cur_schema.parent_idx))) {
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
 * @brief Reads compressed page data to device memory.
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
  cudf::detail::hostdevice_vector<ColumnChunkDesc>& chunks,
  size_t begin_chunk,
  size_t end_chunk,
  std::vector<size_t> const& column_chunk_offsets,
  std::vector<size_type> const& chunk_source_map,
  rmm::cuda_stream_view stream)
{
  // Transfer chunk data, coalescing adjacent chunks
  std::vector<std::future<size_t>> read_tasks;
  for (size_t chunk = begin_chunk; chunk < end_chunk;) {
    size_t const io_offset   = column_chunk_offsets[chunk];
    size_t io_size           = chunks[chunk].compressed_size;
    size_t next_chunk        = chunk + 1;
    bool const is_compressed = (chunks[chunk].codec != Compression::UNCOMPRESSED);
    while (next_chunk < end_chunk) {
      size_t const next_offset      = column_chunk_offsets[next_chunk];
      bool const is_next_compressed = (chunks[next_chunk].codec != Compression::UNCOMPRESSED);
      if (next_offset != io_offset + io_size || is_next_compressed != is_compressed ||
          chunk_source_map[chunk] != chunk_source_map[next_chunk]) {
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
        // Buffer needs to be padded.
        // Required by `gpuDecodePageData`.
        auto buffer =
          rmm::device_buffer(cudf::util::round_up_safe(io_size, BUFFER_PADDING_MULTIPLE), stream);
        auto fut_read_size = source->device_read_async(
          io_offset, io_size, static_cast<uint8_t*>(buffer.data()), stream);
        read_tasks.emplace_back(std::move(fut_read_size));
        page_data[chunk] = datasource::buffer::create(std::move(buffer));
      } else {
        auto const read_buffer = source->host_read(io_offset, io_size);
        // Buffer needs to be padded.
        // Required by `gpuDecodePageData`.
        auto tmp_buffer = rmm::device_buffer(
          cudf::util::round_up_safe(read_buffer->size(), BUFFER_PADDING_MULTIPLE), stream);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          tmp_buffer.data(), read_buffer->data(), read_buffer->size(), cudaMemcpyDefault, stream));
        page_data[chunk] = datasource::buffer::create(std::move(tmp_buffer));
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
[[nodiscard]] size_t count_page_headers(cudf::detail::hostdevice_vector<ColumnChunkDesc>& chunks,
                                        rmm::cuda_stream_view stream)
{
  size_t total_pages = 0;

  kernel_error error_code(stream);
  chunks.host_to_device_async(stream);
  DecodePageHeaders(chunks.device_ptr(), nullptr, chunks.size(), error_code.data(), stream);
  chunks.device_to_host_sync(stream);

  // It's required to ignore unsupported encodings in this function
  // so that we can actually compile a list of all the unsupported encodings found
  // in the pages. That cannot be done here since we do not have the pages vector here.
  // see https://github.com/rapidsai/cudf/pull/14453#pullrequestreview-1778346688
  if (auto const error = error_code.value_sync(stream);
      error != 0 and error != static_cast<uint32_t>(decode_error::UNSUPPORTED_ENCODING)) {
    CUDF_FAIL("Parquet header parsing failed with code(s) while counting page headers " +
              kernel_error::to_string(error));
  }

  for (size_t c = 0; c < chunks.size(); c++) {
    total_pages += chunks[c].num_data_pages + chunks[c].num_dict_pages;
  }

  return total_pages;
}

/**
 * @brief Count the total number of pages using page index information.
 */
[[nodiscard]] size_t count_page_headers_with_pgidx(
  cudf::detail::hostdevice_vector<ColumnChunkDesc>& chunks, rmm::cuda_stream_view stream)
{
  size_t total_pages = 0;
  for (auto& chunk : chunks) {
    CUDF_EXPECTS(chunk.h_chunk_info != nullptr, "Expected non-null column info struct");
    auto const& chunk_info = *chunk.h_chunk_info;
    chunk.num_dict_pages   = chunk_info.has_dictionary() ? 1 : 0;
    chunk.num_data_pages   = chunk_info.pages.size();
    total_pages += chunk.num_data_pages + chunk.num_dict_pages;
  }

  // count_page_headers() also pushes chunks to device, so not using thrust here
  chunks.host_to_device_async(stream);

  return total_pages;
}

// struct used to carry info from the page indexes to the device
struct page_index_info {
  int32_t num_rows;
  int32_t chunk_row;
  int32_t num_nulls;
  int32_t num_valids;
  int32_t str_bytes;
};

// functor to copy page_index_info into the PageInfo struct
struct copy_page_info {
  device_span<page_index_info const> page_indexes;
  device_span<PageInfo> pages;

  __device__ void operator()(size_type idx)
  {
    auto& pg                = pages[idx];
    auto const& pi          = page_indexes[idx];
    pg.num_rows             = pi.num_rows;
    pg.chunk_row            = pi.chunk_row;
    pg.has_page_index       = true;
    pg.num_nulls            = pi.num_nulls;
    pg.num_valids           = pi.num_valids;
    pg.str_bytes_from_index = pi.str_bytes;
    pg.str_bytes            = pi.str_bytes;
    pg.start_val            = 0;
    pg.end_val              = pg.num_valids;
  }
};

/**
 * @brief Set fields on the pages that can be derived from page indexes.
 *
 * This replaces some preprocessing steps, such as page string size calculation.
 */
void fill_in_page_info(host_span<ColumnChunkDesc> chunks,
                       device_span<PageInfo> pages,
                       rmm::cuda_stream_view stream)
{
  auto const num_pages = pages.size();
  std::vector<page_index_info> page_indexes(num_pages);

  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    auto const& chunk = chunks[c];
    CUDF_EXPECTS(chunk.h_chunk_info != nullptr, "Expected non-null column info struct");
    auto const& chunk_info = *chunk.h_chunk_info;
    size_t start_row       = 0;
    page_count += chunk.num_dict_pages;
    for (size_t p = 0; p < chunk_info.pages.size(); p++, page_count++) {
      auto& page      = page_indexes[page_count];
      page.num_rows   = chunk_info.pages[p].num_rows;
      page.chunk_row  = start_row;
      page.num_nulls  = chunk_info.pages[p].num_nulls.value_or(0);
      page.num_valids = chunk_info.pages[p].num_valid.value_or(0);
      page.str_bytes  = chunk_info.pages[p].var_bytes_size.value_or(0);

      start_row += page.num_rows;
    }
  }

  auto d_page_indexes = cudf::detail::make_device_uvector_async(
    page_indexes, stream, rmm::mr::get_current_device_resource());

  auto iter = thrust::make_counting_iterator<size_type>(0);
  thrust::for_each(
    rmm::exec_policy_nosync(stream), iter, iter + num_pages, copy_page_info{d_page_indexes, pages});
}

/**
 * @brief Returns a string representation of known encodings
 *
 * @param encoding Given encoding
 * @return String representation of encoding
 */
std::string encoding_to_string(Encoding encoding)
{
  switch (encoding) {
    case Encoding::PLAIN: return "PLAIN";
    case Encoding::GROUP_VAR_INT: return "GROUP_VAR_INT";
    case Encoding::PLAIN_DICTIONARY: return "PLAIN_DICTIONARY";
    case Encoding::RLE: return "RLE";
    case Encoding::BIT_PACKED: return "BIT_PACKED";
    case Encoding::DELTA_BINARY_PACKED: return "DELTA_BINARY_PACKED";
    case Encoding::DELTA_LENGTH_BYTE_ARRAY: return "DELTA_LENGTH_BYTE_ARRAY";
    case Encoding::DELTA_BYTE_ARRAY: return "DELTA_BYTE_ARRAY";
    case Encoding::RLE_DICTIONARY: return "RLE_DICTIONARY";
    case Encoding::BYTE_STREAM_SPLIT: return "BYTE_STREAM_SPLIT";
    case Encoding::NUM_ENCODINGS:
    default: return "UNKNOWN(" + std::to_string(static_cast<int>(encoding)) + ")";
  }
}

/**
 * @brief Helper function to convert an encoding bitmask to a readable string
 *
 * @param bitmask Bitmask of found unsupported encodings
 * @returns Human readable string with unsupported encodings
 */
[[nodiscard]] std::string encoding_bitmask_to_str(uint32_t encoding_bitmask)
{
  std::bitset<32> bits(encoding_bitmask);
  std::string result;

  for (size_t i = 0; i < bits.size(); ++i) {
    if (bits.test(i)) {
      auto const current = static_cast<Encoding>(i);
      if (!is_supported_encoding(current)) { result.append(encoding_to_string(current) + " "); }
    }
  }
  return result;
}

/**
 * @brief Create a readable string for the user that will list out all unsupported encodings found.
 *
 * @param pages List of page information
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @returns Human readable string with unsupported encodings
 */
[[nodiscard]] std::string list_unsupported_encodings(device_span<PageInfo const> pages,
                                                     rmm::cuda_stream_view stream)
{
  auto const to_mask = [] __device__(auto const& page) {
    return is_supported_encoding(page.encoding) ? 0U : encoding_to_mask(page.encoding);
  };
  uint32_t const unsupported = thrust::transform_reduce(
    rmm::exec_policy(stream), pages.begin(), pages.end(), to_mask, 0U, thrust::bit_or<uint32_t>());
  return encoding_bitmask_to_str(unsupported);
}

/**
 * @brief Sort pages in chunk/schema order
 *
 * @param unsorted_pages The unsorted pages
 * @param chunks The chunks associated with the pages
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @returns The sorted vector of pages
 */
cudf::detail::hostdevice_vector<PageInfo> sort_pages(device_span<PageInfo const> unsorted_pages,
                                                     device_span<ColumnChunkDesc const> chunks,
                                                     rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // sort the pages in chunk/schema order. we use chunk.src_col_index instead of
  // chunk.src_col_schema because the user may have reordered them (reading columns, "a" and "b" but
  // returning them as "b" and "a")
  //
  // ordering of pages is by input column schema, repeated across row groups.  so
  // if we had 3 columns, each with 2 pages, and 1 row group, our schema values might look like
  //
  // 1, 1, 2, 2, 3, 3
  //
  // However, if we had more than one row group, the pattern would be
  //
  // 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3
  // ^ row group 0     |
  //                   ^ row group 1
  //
  // To process pages by key (exclusive_scan_by_key, reduce_by_key, etc), the ordering we actually
  // want is
  //
  // 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
  //
  // We also need to preserve key-relative page ordering, so we need to use a stable sort.
  rmm::device_uvector<int32_t> page_keys{unsorted_pages.size(), stream};
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    unsorted_pages.begin(),
    unsorted_pages.end(),
    page_keys.begin(),
    cuda::proclaim_return_type<int32_t>([chunks = chunks.begin()] __device__(PageInfo const& page) {
      return chunks[page.chunk_idx].src_col_index;
    }));
  // we are doing this by sorting indices first and then transforming the output because nvcc
  // started generating kernels using too much shared memory when trying to sort the pages
  // directly.
  rmm::device_uvector<int32_t> sort_indices(unsorted_pages.size(), stream);
  thrust::sequence(rmm::exec_policy_nosync(stream), sort_indices.begin(), sort_indices.end(), 0);
  thrust::stable_sort_by_key(rmm::exec_policy_nosync(stream),
                             page_keys.begin(),
                             page_keys.end(),
                             sort_indices.begin(),
                             thrust::less<int>());
  auto pass_pages =
    cudf::detail::hostdevice_vector<PageInfo>(unsorted_pages.size(), unsorted_pages.size(), stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    sort_indices.begin(),
    sort_indices.end(),
    pass_pages.d_begin(),
    cuda::proclaim_return_type<PageInfo>([unsorted_pages = unsorted_pages.begin()] __device__(
                                           int32_t i) { return unsorted_pages[i]; }));
  stream.synchronize();
  return pass_pages;
}

/**
 * @brief Decode the page information for a given pass.
 *
 * @param pass_intermediate_data The struct containing pass information
 */
void decode_page_headers(pass_intermediate_data& pass,
                         device_span<PageInfo> unsorted_pages,
                         bool has_page_index,
                         rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  auto iter = thrust::make_counting_iterator(0);
  rmm::device_uvector<size_t> chunk_page_counts(pass.chunks.size() + 1, stream);
  thrust::transform_exclusive_scan(
    rmm::exec_policy_nosync(stream),
    iter,
    iter + pass.chunks.size() + 1,
    chunk_page_counts.begin(),
    cuda::proclaim_return_type<size_t>(
      [chunks = pass.chunks.d_begin(), num_chunks = pass.chunks.size()] __device__(size_t i) {
        return static_cast<size_t>(
          i >= num_chunks ? 0 : chunks[i].num_data_pages + chunks[i].num_dict_pages);
      }),
    0,
    thrust::plus<size_t>{});
  rmm::device_uvector<chunk_page_info> d_chunk_page_info(pass.chunks.size(), stream);
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   iter,
                   iter + pass.chunks.size(),
                   [cpi               = d_chunk_page_info.begin(),
                    chunk_page_counts = chunk_page_counts.begin(),
                    unsorted_pages    = unsorted_pages.begin()] __device__(size_t i) {
                     cpi[i].pages = &unsorted_pages[chunk_page_counts[i]];
                   });

  kernel_error error_code(stream);
  DecodePageHeaders(pass.chunks.d_begin(),
                    d_chunk_page_info.begin(),
                    pass.chunks.size(),
                    error_code.data(),
                    stream);

  if (auto const error = error_code.value_sync(stream); error != 0) {
    if (BitAnd(error, decode_error::UNSUPPORTED_ENCODING) != 0) {
      auto const unsupported_str =
        ". With unsupported encodings found: " + list_unsupported_encodings(pass.pages, stream);
      CUDF_FAIL("Parquet header parsing failed with code(s) " + kernel_error::to_string(error) +
                unsupported_str);
    } else {
      CUDF_FAIL("Parquet header parsing failed with code(s) " + kernel_error::to_string(error));
    }
  }

  if (has_page_index) { fill_in_page_info(pass.chunks, unsorted_pages, stream); }

  // compute max bytes needed for level data
  auto level_bit_size = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<int>([chunks = pass.chunks.d_begin()] __device__(int i) {
      auto c = chunks[i];
      return static_cast<int>(
        max(c.level_bits[level_type::REPETITION], c.level_bits[level_type::DEFINITION]));
    }));
  // max level data bit size.
  int const max_level_bits = thrust::reduce(rmm::exec_policy(stream),
                                            level_bit_size,
                                            level_bit_size + pass.chunks.size(),
                                            0,
                                            thrust::maximum<int>());
  pass.level_type_size     = std::max(1, cudf::util::div_rounding_up_safe(max_level_bits, 8));

  // sort the pages in chunk/schema order.
  pass.pages = sort_pages(unsorted_pages, pass.chunks, stream);

  // compute offsets to each group of input pages.
  // page_keys:   1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
  //
  // result:      0,          4,          8
  rmm::device_uvector<size_type> page_counts(pass.pages.size() + 1, stream);
  auto page_keys             = make_page_key_iterator(pass.pages);
  auto const page_counts_end = thrust::reduce_by_key(rmm::exec_policy(stream),
                                                     page_keys,
                                                     page_keys + pass.pages.size(),
                                                     thrust::make_constant_iterator(1),
                                                     thrust::make_discard_iterator(),
                                                     page_counts.begin())
                                 .second;
  auto const num_page_counts = page_counts_end - page_counts.begin();
  pass.page_offsets          = rmm::device_uvector<size_type>(num_page_counts + 1, stream);
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         page_counts.begin(),
                         page_counts.begin() + num_page_counts + 1,
                         pass.page_offsets.begin());

  // setup dict_page for each chunk if necessary
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   pass.pages.d_begin(),
                   pass.pages.d_end(),
                   [chunks = pass.chunks.d_begin()] __device__(PageInfo const& p) {
                     if (p.flags & PAGEINFO_FLAGS_DICTIONARY) {
                       chunks[p.chunk_idx].dict_page = &p;
                     }
                   });

  pass.pages.device_to_host_async(stream);
  pass.chunks.device_to_host_async(stream);
  stream.synchronize();
}

constexpr bool is_string_chunk(ColumnChunkDesc const& chunk)
{
  auto const is_decimal =
    chunk.logical_type.has_value() and chunk.logical_type->type == LogicalType::DECIMAL;
  auto const is_binary =
    chunk.physical_type == BYTE_ARRAY or chunk.physical_type == FIXED_LEN_BYTE_ARRAY;
  return is_binary and not is_decimal;
}

struct set_str_dict_index_count {
  device_span<size_t> str_dict_index_count;
  device_span<const ColumnChunkDesc> chunks;

  __device__ void operator()(PageInfo const& page)
  {
    auto const& chunk = chunks[page.chunk_idx];
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0 and chunk.num_dict_pages > 0 and
        is_string_chunk(chunk)) {
      // there is only ever one dictionary page per chunk, so this is safe to do in parallel.
      str_dict_index_count[page.chunk_idx] = page.num_input_values;
    }
  }
};

struct set_str_dict_index_ptr {
  string_index_pair* const base;
  device_span<const size_t> str_dict_index_offsets;
  device_span<ColumnChunkDesc> chunks;

  __device__ void operator()(size_t i)
  {
    auto& chunk = chunks[i];
    if (chunk.num_dict_pages > 0 and is_string_chunk(chunk)) {
      chunk.str_dict_index = base + str_dict_index_offsets[i];
    }
  }
};

/**
 * @brief Functor which computes an estimated row count for list pages.
 *
 */
struct set_list_row_count_estimate {
  device_span<const ColumnChunkDesc> chunks;

  __device__ void operator()(PageInfo& page)
  {
    if (page.flags & PAGEINFO_FLAGS_DICTIONARY) { return; }
    auto const& chunk  = chunks[page.chunk_idx];
    auto const is_list = chunk.max_level[level_type::REPETITION] > 0;
    if (!is_list) { return; }

    // For LIST pages that we have not yet decoded, page.num_rows is not an accurate number.
    // so we instead estimate the number of rows as follows:
    // - each chunk stores an estimated number of bytes per row E
    // - estimate number of rows in a page = page.uncompressed_page_size / E
    //
    // it is not required that this number is accurate. we just want it to be somewhat close so that
    // we get reasonable results as we choose subpass splits.
    //
    // all other columns can use page.num_rows directly as it will be accurate.
    page.num_rows = static_cast<size_t>(static_cast<float>(page.uncompressed_page_size) /
                                        chunk.list_bytes_per_row_est);
  }
};

/**
 * @brief Set the expected row count on the final page for all columns.
 *
 */
struct set_final_row_count {
  device_span<PageInfo> pages;
  device_span<const ColumnChunkDesc> chunks;

  __device__ void operator()(size_t i)
  {
    auto& page        = pages[i];
    auto const& chunk = chunks[page.chunk_idx];
    // only do this for the last page in each chunk
    if (i < pages.size() - 1 && (pages[i + 1].chunk_idx == page.chunk_idx)) { return; }
    size_t const page_start_row = chunk.start_row + page.chunk_row;
    size_t const chunk_last_row = chunk.start_row + chunk.num_rows;
    page.num_rows               = chunk_last_row - page_start_row;
  }
};

}  // anonymous namespace

void reader::impl::build_string_dict_indices()
{
  CUDF_FUNC_RANGE();

  auto& pass = *_pass_itm_data;

  // compute number of indices per chunk and a summed total
  rmm::device_uvector<size_t> str_dict_index_count(pass.chunks.size() + 1, _stream);
  thrust::fill(
    rmm::exec_policy_nosync(_stream), str_dict_index_count.begin(), str_dict_index_count.end(), 0);
  thrust::for_each(rmm::exec_policy_nosync(_stream),
                   pass.pages.d_begin(),
                   pass.pages.d_end(),
                   set_str_dict_index_count{str_dict_index_count, pass.chunks});

  size_t const total_str_dict_indexes = thrust::reduce(
    rmm::exec_policy(_stream), str_dict_index_count.begin(), str_dict_index_count.end());
  if (total_str_dict_indexes == 0) { return; }

  // convert to offsets
  rmm::device_uvector<size_t>& str_dict_index_offsets = str_dict_index_count;
  thrust::exclusive_scan(rmm::exec_policy_nosync(_stream),
                         str_dict_index_offsets.begin(),
                         str_dict_index_offsets.end(),
                         str_dict_index_offsets.begin(),
                         0);

  // allocate and distribute pointers
  pass.str_dict_index = cudf::detail::make_zeroed_device_uvector_async<string_index_pair>(
    total_str_dict_indexes, _stream, rmm::mr::get_current_device_resource());

  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(
    rmm::exec_policy_nosync(_stream),
    iter,
    iter + pass.chunks.size(),
    set_str_dict_index_ptr{pass.str_dict_index.data(), str_dict_index_offsets, pass.chunks});

  // compute the indices
  BuildStringDictionaryIndex(pass.chunks.device_ptr(), pass.chunks.size(), _stream);
  pass.chunks.device_to_host_sync(_stream);
}

void reader::impl::allocate_nesting_info()
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  auto const num_columns         = _input_columns.size();
  auto& pages                    = subpass.pages;
  auto& page_nesting_info        = subpass.page_nesting_info;
  auto& page_nesting_decode_info = subpass.page_nesting_decode_info;

  // generate the number of nesting info structs needed per-page, by column
  std::vector<int> per_page_nesting_info_size(num_columns);
  auto iter = thrust::make_counting_iterator(size_type{0});
  std::transform(iter, iter + num_columns, per_page_nesting_info_size.begin(), [&](size_type i) {
    auto const schema_idx = _input_columns[i].schema_idx;
    auto const& schema    = _metadata->get_schema(schema_idx);
    return max(schema.max_definition_level + 1, _metadata->get_output_nesting_depth(schema_idx));
  });

  // compute total # of page_nesting infos needed and allocate space. doing this in one
  // buffer to keep it to a single gpu allocation
  auto counting_iter = thrust::make_counting_iterator(size_t{0});
  size_t const total_page_nesting_infos =
    std::accumulate(counting_iter, counting_iter + num_columns, 0, [&](int total, size_t index) {
      return total + (per_page_nesting_info_size[index] * subpass.column_page_count[index]);
    });

  page_nesting_info =
    cudf::detail::hostdevice_vector<PageNestingInfo>{total_page_nesting_infos, _stream};
  page_nesting_decode_info =
    cudf::detail::hostdevice_vector<PageNestingDecodeInfo>{total_page_nesting_infos, _stream};

  // update pointers in the PageInfos
  int target_page_index = 0;
  int src_info_index    = 0;
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    auto const src_col_schema = _input_columns[idx].schema_idx;

    for (size_t p_idx = 0; p_idx < subpass.column_page_count[idx]; p_idx++) {
      pages[target_page_index + p_idx].nesting = page_nesting_info.device_ptr() + src_info_index;
      pages[target_page_index + p_idx].nesting_decode =
        page_nesting_decode_info.device_ptr() + src_info_index;

      pages[target_page_index + p_idx].nesting_info_size = per_page_nesting_info_size[idx];
      pages[target_page_index + p_idx].num_output_nesting_levels =
        _metadata->get_output_nesting_depth(src_col_schema);

      src_info_index += per_page_nesting_info_size[idx];
    }
    target_page_index += subpass.column_page_count[idx];
  }

  // fill in
  int nesting_info_index = 0;
  std::map<int, std::pair<std::vector<int>, std::vector<int>>> depth_remapping;
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    auto const src_col_schema = _input_columns[idx].schema_idx;

    // schema of the input column
    auto& schema = _metadata->get_schema(src_col_schema);
    // real depth of the output cudf column hierarchy (1 == no nesting, 2 == 1 level, etc)
    int const max_output_depth = _metadata->get_output_nesting_depth(src_col_schema);

    // if this column has lists, generate depth remapping
    std::map<int, std::pair<std::vector<int>, std::vector<int>>> depth_remapping;
    if (schema.max_repetition_level > 0) {
      generate_depth_remappings(depth_remapping, src_col_schema, *_metadata);
    }

    // fill in host-side nesting info
    int schema_idx  = src_col_schema;
    auto cur_schema = _metadata->get_schema(schema_idx);
    int cur_depth   = max_output_depth - 1;
    while (schema_idx > 0) {
      // stub columns (basically the inner field of a list schema element) are not real columns.
      // we can ignore them for the purposes of output nesting info
      if (!cur_schema.is_stub()) {
        // initialize each page within the chunk
        for (size_t p_idx = 0; p_idx < subpass.column_page_count[idx]; p_idx++) {
          PageNestingInfo* pni =
            &page_nesting_info[nesting_info_index + (p_idx * per_page_nesting_info_size[idx])];

          PageNestingDecodeInfo* nesting_info =
            &page_nesting_decode_info[nesting_info_index +
                                      (p_idx * per_page_nesting_info_size[idx])];

          // if we have lists, set our start and end depth remappings
          if (schema.max_repetition_level > 0) {
            auto remap = depth_remapping.find(src_col_schema);
            CUDF_EXPECTS(remap != depth_remapping.end(),
                         "Could not find depth remapping for schema");
            std::vector<int> const& rep_depth_remap = (remap->second.first);
            std::vector<int> const& def_depth_remap = (remap->second.second);

            for (size_t m = 0; m < rep_depth_remap.size(); m++) {
              nesting_info[m].start_depth = rep_depth_remap[m];
            }
            for (size_t m = 0; m < def_depth_remap.size(); m++) {
              nesting_info[m].end_depth = def_depth_remap[m];
            }
          }

          // values indexed by output column index
          nesting_info[cur_depth].max_def_level = cur_schema.max_definition_level;
          pni[cur_depth].size                   = 0;
          pni[cur_depth].type =
            to_type_id(cur_schema, _strings_to_categorical, _options.timestamp_type.id());
          pni[cur_depth].nullable = cur_schema.repetition_type == OPTIONAL;
        }

        // move up the hierarchy
        cur_depth--;
      }

      // next schema
      schema_idx = cur_schema.parent_idx;
      cur_schema = _metadata->get_schema(schema_idx);
    }

    nesting_info_index += (per_page_nesting_info_size[idx] * subpass.column_page_count[idx]);
  }

  // copy nesting info to the device
  page_nesting_info.host_to_device_async(_stream);
  page_nesting_decode_info.host_to_device_async(_stream);
}

void reader::impl::allocate_level_decode_space()
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  auto& pages = subpass.pages;

  // TODO: this could be made smaller if we ignored dictionary pages and pages with no
  // repetition data.
  size_t const per_page_decode_buf_size = LEVEL_DECODE_BUF_SIZE * 2 * pass.level_type_size;
  auto const decode_buf_size            = per_page_decode_buf_size * pages.size();
  subpass.level_decode_data =
    rmm::device_buffer(decode_buf_size, _stream, rmm::mr::get_current_device_resource());

  // distribute the buffers
  uint8_t* buf = static_cast<uint8_t*>(subpass.level_decode_data.data());
  for (size_t idx = 0; idx < pages.size(); idx++) {
    auto& p = pages[idx];

    p.lvl_decode_buf[level_type::DEFINITION] = buf;
    buf += (LEVEL_DECODE_BUF_SIZE * pass.level_type_size);
    p.lvl_decode_buf[level_type::REPETITION] = buf;
    buf += (LEVEL_DECODE_BUF_SIZE * pass.level_type_size);
  }
}

std::pair<bool, std::vector<std::future<void>>> reader::impl::read_column_chunks()
{
  auto const& row_groups_info = _pass_itm_data->row_groups;

  auto& raw_page_data = _pass_itm_data->raw_page_data;
  auto& chunks        = _pass_itm_data->chunks;

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_chunks        = row_groups_info.size() * num_input_columns;

  // Association between each column chunk and its source
  std::vector<size_type> chunk_source_map(num_chunks);

  // Tracker for eventually deallocating compressed and uncompressed data
  raw_page_data = std::vector<std::unique_ptr<datasource::buffer>>(num_chunks);

  // Keep track of column chunk file offsets
  std::vector<size_t> column_chunk_offsets(num_chunks);

  // Initialize column chunk information
  size_t total_decompressed_size = 0;
  // TODO: make this respect the pass-wide skip_rows/num_rows instead of the file-wide
  // skip_rows/num_rows
  // auto remaining_rows            = num_rows;
  std::vector<std::future<void>> read_chunk_tasks;
  size_type chunk_count = 0;
  for (auto const& rg : row_groups_info) {
    auto const& row_group       = _metadata->get_row_group(rg.index, rg.source_index);
    auto const row_group_source = rg.source_index;

    // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
    for (size_t i = 0; i < num_input_columns; ++i) {
      auto const& col = _input_columns[i];
      // look up metadata
      auto& col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);

      column_chunk_offsets[chunk_count] =
        (col_meta.dictionary_page_offset != 0)
          ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
          : col_meta.data_page_offset;

      // Map each column chunk to its column index and its source index
      chunk_source_map[chunk_count] = row_group_source;

      if (col_meta.codec != Compression::UNCOMPRESSED) {
        total_decompressed_size += col_meta.total_uncompressed_size;
      }

      chunk_count++;
    }
  }

  // Read compressed chunk data to device memory
  read_chunk_tasks.push_back(read_column_chunks_async(_sources,
                                                      raw_page_data,
                                                      chunks,
                                                      0,
                                                      chunks.size(),
                                                      column_chunk_offsets,
                                                      chunk_source_map,
                                                      _stream));

  return {total_decompressed_size > 0, std::move(read_chunk_tasks)};
}

void reader::impl::read_compressed_data()
{
  auto& pass = *_pass_itm_data;

  // This function should never be called if `num_rows == 0`.
  CUDF_EXPECTS(_pass_itm_data->num_rows > 0, "Number of reading rows must not be zero.");

  auto& chunks = pass.chunks;

  auto const [has_compressed_data, read_chunks_tasks] = read_column_chunks();
  pass.has_compressed_data                            = has_compressed_data;

  for (auto& task : read_chunks_tasks) {
    task.wait();
  }

  // Process dataset chunk pages into output columns
  auto const total_pages = _has_page_index ? count_page_headers_with_pgidx(chunks, _stream)
                                           : count_page_headers(chunks, _stream);
  if (total_pages <= 0) { return; }
  rmm::device_uvector<PageInfo> unsorted_pages(total_pages, _stream);

  // decoding of column/page information
  decode_page_headers(pass, unsorted_pages, _has_page_index, _stream);
  CUDF_EXPECTS(pass.page_offsets.size() - 1 == static_cast<size_t>(_input_columns.size()),
               "Encountered page_offsets / num_columns mismatch");
}

namespace {

struct cumulative_row_info {
  size_t row_count;   // cumulative row count
  size_t size_bytes;  // cumulative size in bytes
  int key;            // schema index
};

struct get_page_chunk_idx {
  __device__ size_type operator()(PageInfo const& page) { return page.chunk_idx; }
};

struct get_page_num_rows {
  __device__ size_type operator()(PageInfo const& page) { return page.num_rows; }
};

struct input_col_info {
  int const schema_idx;
  size_type const nesting_depth;
};

/**
 * @brief Converts a 1-dimensional index into page, depth and column indices used in
 * allocate_columns to compute columns sizes.
 *
 * The input index will iterate through pages, nesting depth and column indices in that order.
 */
struct reduction_indices {
  size_t const page_idx;
  size_type const depth_idx;
  size_type const col_idx;

  __device__ reduction_indices(size_t index_, size_type max_depth_, size_t num_pages_)
    : page_idx(index_ % num_pages_),
      depth_idx((index_ / num_pages_) % max_depth_),
      col_idx(index_ / (max_depth_ * num_pages_))
  {
  }
};

/**
 * @brief Returns the size field of a PageInfo struct for a given depth, keyed by schema.
 */
struct get_page_nesting_size {
  input_col_info const* const input_cols;
  size_type const max_depth;
  size_t const num_pages;
  PageInfo const* const pages;

  __device__ size_type operator()(size_t index) const
  {
    auto const indices = reduction_indices{index, max_depth, num_pages};

    auto const& page = pages[indices.page_idx];
    if (page.src_col_schema != input_cols[indices.col_idx].schema_idx ||
        page.flags & PAGEINFO_FLAGS_DICTIONARY ||
        indices.depth_idx >= input_cols[indices.col_idx].nesting_depth) {
      return 0;
    }

    return page.nesting[indices.depth_idx].batch_size;
  }
};

struct get_reduction_key {
  size_t const num_pages;
  __device__ size_t operator()(size_t index) const { return index / num_pages; }
};

/**
 * @brief Writes to the chunk_row field of the PageInfo struct.
 */
struct chunk_row_output_iter {
  PageInfo* p;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  __host__ __device__ chunk_row_output_iter operator+(int i) { return {p + i}; }

  __host__ __device__ chunk_row_output_iter& operator++()
  {
    p++;
    return *this;
  }

  __device__ reference operator[](int i) { return p[i].chunk_row; }
  __device__ reference operator*() { return p->chunk_row; }
};

/**
 * @brief Writes to the page_start_value field of the PageNestingInfo struct, keyed by schema.
 */
/**
 * @brief Writes to the page_start_value field of the PageNestingInfo struct, keyed by schema.
 */
struct start_offset_output_iterator {
  PageInfo const* pages;
  size_t cur_index;
  input_col_info const* input_cols;
  size_type max_depth;
  size_t num_pages;
  int empty               = 0;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  constexpr void operator=(start_offset_output_iterator const& other)
  {
    pages      = other.pages;
    cur_index  = other.cur_index;
    input_cols = other.input_cols;
    max_depth  = other.max_depth;
    num_pages  = other.num_pages;
  }

  constexpr start_offset_output_iterator operator+(size_t i)
  {
    return start_offset_output_iterator{pages, cur_index + i, input_cols, max_depth, num_pages};
  }

  constexpr start_offset_output_iterator& operator++()
  {
    cur_index++;
    return *this;
  }

  __device__ reference operator[](size_t i) { return dereference(cur_index + i); }
  __device__ reference operator*() { return dereference(cur_index); }

 private:
  __device__ reference dereference(size_t index)
  {
    auto const indices = reduction_indices{index, max_depth, num_pages};

    PageInfo const& p = pages[indices.page_idx];
    if (p.src_col_schema != input_cols[indices.col_idx].schema_idx ||
        p.flags & PAGEINFO_FLAGS_DICTIONARY ||
        indices.depth_idx >= input_cols[indices.col_idx].nesting_depth) {
      return empty;
    }
    return p.nesting_decode[indices.depth_idx].page_start_value;
  }
};

struct page_to_string_size {
  ColumnChunkDesc const* chunks;

  __device__ size_t operator()(PageInfo const& page) const
  {
    auto const chunk = chunks[page.chunk_idx];

    if (not is_string_col(chunk) || (page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { return 0; }
    return page.str_bytes;
  }
};

struct page_offset_output_iter {
  PageInfo* p;

  using value_type        = size_t;
  using difference_type   = size_t;
  using pointer           = size_t*;
  using reference         = size_t&;
  using iterator_category = thrust::output_device_iterator_tag;

  __host__ __device__ page_offset_output_iter operator+(int i) { return {p + i}; }

  __host__ __device__ page_offset_output_iter& operator++()
  {
    p++;
    return *this;
  }

  __device__ reference operator[](int i) { return p[i].str_offset; }
  __device__ reference operator*() { return p->str_offset; }
};
// update chunk_row field in subpass page from pass page
struct update_subpass_chunk_row {
  device_span<PageInfo> pass_pages;
  device_span<PageInfo> subpass_pages;
  device_span<size_t> page_src_index;

  __device__ void operator()(size_t i)
  {
    subpass_pages[i].chunk_row = pass_pages[page_src_index[i]].chunk_row;
  }
};

// update num_rows field from pass page to subpass page
struct update_pass_num_rows {
  device_span<PageInfo> pass_pages;
  device_span<PageInfo> subpass_pages;
  device_span<size_t> page_src_index;

  __device__ void operator()(size_t i)
  {
    pass_pages[page_src_index[i]].num_rows = subpass_pages[i].num_rows;
  }
};

}  // anonymous namespace

void reader::impl::preprocess_file(read_mode mode)
{
  CUDF_EXPECTS(!_file_preprocessed, "Attempted to preprocess file more than once");

  // if filter is not empty, then create output types as vector and pass for filtering.

  std::vector<data_type> output_dtypes;
  if (_expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  std::tie(
    _file_itm_data.global_skip_rows, _file_itm_data.global_num_rows, _file_itm_data.row_groups) =
    _metadata->select_row_groups(_options.row_group_indices,
                                 _options.skip_rows,
                                 _options.num_rows,
                                 output_dtypes,
                                 _output_column_schemas,
                                 _expr_conv.get_converted_expr(),
                                 _stream);

  // check for page indexes
  _has_page_index = std::all_of(_file_itm_data.row_groups.begin(),
                                _file_itm_data.row_groups.end(),
                                [](auto const& row_group) { return row_group.has_page_index(); });

  if (_file_itm_data.global_num_rows > 0 && not _file_itm_data.row_groups.empty() &&
      not _input_columns.empty()) {
    // fills in chunk information without physically loading or decompressing
    // the associated data
    create_global_chunk_info();

    // compute schedule of input reads.
    compute_input_passes();
  }

#if defined(PARQUET_CHUNK_LOGGING)
  printf("==============================================\n");
  setlocale(LC_NUMERIC, "");
  printf("File: skip_rows(%'lu), num_rows(%'lu), input_read_limit(%'lu), output_read_limit(%'lu)\n",
         _file_itm_data.global_skip_rows,
         _file_itm_data.global_num_rows,
         _input_pass_read_limit,
         _output_chunk_read_limit);
  printf("# Row groups: %'lu\n", _file_itm_data.row_groups.size());
  printf("# Input passes: %'lu\n", _file_itm_data.num_passes());
  printf("# Input columns: %'lu\n", _input_columns.size());
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    auto const& schema = _metadata->get_schema(_input_columns[idx].schema_idx);
    auto const type_id = to_type_id(schema, _strings_to_categorical, _options.timestamp_type.id());
    printf("\tC(%'lu, %s): %s\n",
           idx,
           _input_columns[idx].name.c_str(),
           cudf::type_to_name(cudf::data_type{type_id}).c_str());
  }
  printf("# Output columns: %'lu\n", _output_buffers.size());
  for (size_t idx = 0; idx < _output_buffers.size(); idx++) {
    printf("\tC(%'lu): %s\n", idx, cudf::io::detail::type_to_name(_output_buffers[idx]).c_str());
  }
#endif

  _file_preprocessed = true;
}

void reader::impl::generate_list_column_row_count_estimates()
{
  auto& pass = *_pass_itm_data;
  thrust::for_each(rmm::exec_policy(_stream),
                   pass.pages.d_begin(),
                   pass.pages.d_end(),
                   set_list_row_count_estimate{pass.chunks});

  // computes:
  // PageInfo::chunk_row (the chunk-relative row index) for all pages in the pass. The start_row
  // field in ColumnChunkDesc is the absolute row index for the whole file. chunk_row in PageInfo is
  // relative to the beginning of the chunk. so in the kernels, chunk.start_row + page.chunk_row
  // gives us the absolute row index
  // Note: chunk_row is already computed if we have column indexes
  if (not _has_page_index) {
    auto key_input  = thrust::make_transform_iterator(pass.pages.d_begin(), get_page_chunk_idx{});
    auto page_input = thrust::make_transform_iterator(pass.pages.d_begin(), get_page_num_rows{});
    thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(_stream),
                                  key_input,
                                  key_input + pass.pages.size(),
                                  page_input,
                                  chunk_row_output_iter{pass.pages.device_ptr()});
  }

  // to compensate for the list row size estimates, force the row count on the last page for each
  // column chunk (each rowgroup) such that it ends on the real known row count. this is so that as
  // we march through the subpasses, we will find that every column cleanly ends up the expected row
  // count at the row group boundary and our split computations work correctly.
  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(rmm::exec_policy_nosync(_stream),
                   iter,
                   iter + pass.pages.size(),
                   set_final_row_count{pass.pages, pass.chunks});

  pass.chunks.device_to_host_async(_stream);
  pass.pages.device_to_host_async(_stream);
  _stream.synchronize();
}

void reader::impl::preprocess_subpass_pages(read_mode mode, size_t chunk_read_limit)
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  // iterate over all input columns and determine if they contain lists.
  // TODO: we could do this once at the file level instead of every time we get in here. the set of
  // columns we are processing does not change over multiple passes/subpasses/output chunks.
  bool has_lists = false;
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    auto const& input_col  = _input_columns[idx];
    size_t const max_depth = input_col.nesting_depth();

    auto* cols = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // if this has a list parent, we have to get column sizes from the
      // data computed during ComputePageSizes
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) {
        has_lists = true;
        break;
      }
    }
    if (has_lists) { break; }
  }

  // in some cases we will need to do further preprocessing of pages.
  // - if we have lists, the num_rows field in PageInfo will be incorrect coming out of the file
  // - if we are doing a chunked read, we need to compute the size of all string data
  if (has_lists || chunk_read_limit > 0) {
    // computes:
    // PageNestingInfo::num_rows for each page. the true number of rows (taking repetition into
    // account), not just the number of values. PageNestingInfo::size for each level of nesting, for
    // each page.
    //
    // we will be applying a later "trim" pass if skip_rows/num_rows is being used, which can happen
    // if:
    // - user has passed custom row bounds
    // - we will be doing a chunked read
    ComputePageSizes(subpass.pages,
                     pass.chunks,
                     0,  // 0-max size_t. process all possible rows
                     std::numeric_limits<size_t>::max(),
                     true,                  // compute num_rows
                     chunk_read_limit > 0,  // compute string sizes
                     _pass_itm_data->level_type_size,
                     _stream);
  }

  auto iter = thrust::make_counting_iterator(0);

  // copy our now-correct row counts  back to the base pages stored in the pass.
  // only need to do this if we are not processing the whole pass in one subpass
  if (!subpass.single_subpass) {
    thrust::for_each(rmm::exec_policy_nosync(_stream),
                     iter,
                     iter + subpass.pages.size(),
                     update_pass_num_rows{pass.pages, subpass.pages, subpass.page_src_index});
  }

  // computes:
  // PageInfo::chunk_row (the chunk-relative row index) for all pages in the pass. The start_row
  // field in ColumnChunkDesc is the absolute row index for the whole file. chunk_row in PageInfo is
  // relative to the beginning of the chunk. so in the kernels, chunk.start_row + page.chunk_row
  // gives us the absolute row index
  auto key_input  = thrust::make_transform_iterator(pass.pages.d_begin(), get_page_chunk_idx{});
  auto page_input = thrust::make_transform_iterator(pass.pages.d_begin(), get_page_num_rows{});
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(_stream),
                                key_input,
                                key_input + pass.pages.size(),
                                page_input,
                                chunk_row_output_iter{pass.pages.device_ptr()});

  // copy chunk row into the subpass pages
  // only need to do this if we are not processing the whole pass in one subpass
  if (!subpass.single_subpass) {
    thrust::for_each(rmm::exec_policy_nosync(_stream),
                     iter,
                     iter + subpass.pages.size(),
                     update_subpass_chunk_row{pass.pages, subpass.pages, subpass.page_src_index});
  }

  // retrieve pages back
  pass.pages.device_to_host_async(_stream);
  if (!subpass.single_subpass) { subpass.pages.device_to_host_async(_stream); }
  _stream.synchronize();

  // at this point we have an accurate row count so we can compute how many rows we will actually be
  // able to decode for this pass. we will have selected a set of pages for each column in the
  // row group, but not every page will have the same number of rows. so, we can only read as many
  // rows as the smallest batch (by column) we have decompressed.
  size_t page_index = 0;
  size_t max_row    = std::numeric_limits<size_t>::max();
  auto const last_pass_row =
    _file_itm_data.input_pass_start_row_count[_file_itm_data._current_input_pass + 1];
  for (size_t idx = 0; idx < subpass.column_page_count.size(); idx++) {
    auto const& last_page = subpass.pages[page_index + (subpass.column_page_count[idx] - 1)];
    auto const& chunk     = pass.chunks[last_page.chunk_idx];

    size_t max_col_row =
      static_cast<size_t>(chunk.start_row + last_page.chunk_row + last_page.num_rows);
    // special case.  list rows can span page boundaries, but we can't tell if that is happening
    // here because we have not yet decoded the pages. the very last row starting in the page may
    // not terminate in the page. to handle this, only decode up to the second to last row in the
    // subpass since we know that will safely completed.
    bool const is_list = chunk.max_level[level_type::REPETITION] > 0;
    if (is_list && max_col_row < last_pass_row) {
      size_t const min_col_row = static_cast<size_t>(chunk.start_row + last_page.chunk_row);
      CUDF_EXPECTS((max_col_row - min_col_row) > 1, "Unexpected short subpass");
      max_col_row--;
    }

    max_row = min(max_row, max_col_row);

    page_index += subpass.column_page_count[idx];
  }
  subpass.skip_rows   = pass.skip_rows + pass.processed_rows;
  auto const pass_end = pass.skip_rows + pass.num_rows;
  max_row             = min(max_row, pass_end);
  subpass.num_rows    = max_row - subpass.skip_rows;

  // now split up the output into chunks as necessary
  compute_output_chunks_for_subpass();
}

void reader::impl::allocate_columns(read_mode mode, size_t skip_rows, size_t num_rows)
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  // Should not reach here if there is no page data.
  CUDF_EXPECTS(subpass.pages.size() > 0, "There are no pages present in the subpass");

  // computes:
  // PageNestingInfo::batch_size for each level of nesting, for each page, taking row bounds into
  // account. PageInfo::skipped_values, which tells us where to start decoding in the input to
  // respect the user bounds. It is only necessary to do this second pass if uses_custom_row_bounds
  // is set (if the user has specified artificial bounds).
  if (uses_custom_row_bounds(mode)) {
    ComputePageSizes(subpass.pages,
                     pass.chunks,
                     skip_rows,
                     num_rows,
                     false,  // num_rows is already computed
                     false,  // no need to compute string sizes
                     pass.level_type_size,
                     _stream);
  }

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

      // if this has a list parent, we have to get column sizes from the
      // data computed during ComputePageSizes
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) {
        has_lists = true;
      }
      // if we haven't already processed this column because it is part of a struct hierarchy
      else if (out_buf.size == 0) {
        // add 1 for the offset if this is a list column
        // we're going to start null mask as all valid and then turn bits off if necessary
        out_buf.create_with_mask(
          out_buf.type.id() == type_id::LIST && l_idx < max_depth ? num_rows + 1 : num_rows,
          cudf::mask_state::ALL_VALID,
          _stream,
          _mr);
      }
    }
  }

  // compute output column sizes by examining the pages of the -input- columns
  if (has_lists) {
    std::vector<input_col_info> h_cols_info;
    h_cols_info.reserve(_input_columns.size());
    std::transform(_input_columns.cbegin(),
                   _input_columns.cend(),
                   std::back_inserter(h_cols_info),
                   [](auto& col) -> input_col_info {
                     return {col.schema_idx, static_cast<size_type>(col.nesting_depth())};
                   });

    auto const max_depth =
      (*std::max_element(h_cols_info.cbegin(),
                         h_cols_info.cend(),
                         [](auto& l, auto& r) { return l.nesting_depth < r.nesting_depth; }))
        .nesting_depth;

    auto const d_cols_info = cudf::detail::make_device_uvector_async(
      h_cols_info, _stream, rmm::mr::get_current_device_resource());

    auto const num_keys = _input_columns.size() * max_depth * subpass.pages.size();
    // size iterator. indexes pages by sorted order
    rmm::device_uvector<size_type> size_input{num_keys, _stream};
    thrust::transform(
      rmm::exec_policy(_stream),
      thrust::make_counting_iterator<size_type>(0),
      thrust::make_counting_iterator<size_type>(num_keys),
      size_input.begin(),
      get_page_nesting_size{
        d_cols_info.data(), max_depth, subpass.pages.size(), subpass.pages.device_begin()});
    auto const reduction_keys =
      cudf::detail::make_counting_transform_iterator(0, get_reduction_key{subpass.pages.size()});
    cudf::detail::hostdevice_vector<size_t> sizes{_input_columns.size() * max_depth, _stream};

    // find the size of each column
    thrust::reduce_by_key(rmm::exec_policy(_stream),
                          reduction_keys,
                          reduction_keys + num_keys,
                          size_input.cbegin(),
                          thrust::make_discard_iterator(),
                          sizes.d_begin());

    // for nested hierarchies, compute per-page start offset
    thrust::exclusive_scan_by_key(
      rmm::exec_policy(_stream),
      reduction_keys,
      reduction_keys + num_keys,
      size_input.cbegin(),
      start_offset_output_iterator{
        subpass.pages.device_begin(), 0, d_cols_info.data(), max_depth, subpass.pages.size()});

    sizes.device_to_host_sync(_stream);
    for (size_type idx = 0; idx < static_cast<size_type>(_input_columns.size()); idx++) {
      auto const& input_col = _input_columns[idx];
      auto* cols            = &_output_buffers;
      for (size_type l_idx = 0; l_idx < static_cast<size_type>(input_col.nesting_depth());
           l_idx++) {
        auto& out_buf = (*cols)[input_col.nesting[l_idx]];
        cols          = &out_buf.children;
        // if this buffer is part of a list hierarchy, we need to determine it's
        // final size and allocate it here.
        //
        // for struct columns, higher levels of the output columns are shared between input
        // columns. so don't compute any given level more than once.
        if ((out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) && out_buf.size == 0) {
          auto size = sizes[(idx * max_depth) + l_idx];

          // if this is a list column add 1 for non-leaf levels for the terminating offset
          if (out_buf.type.id() == type_id::LIST && l_idx < max_depth) { size++; }

          // allocate
          // we're going to start null mask as all valid and then turn bits off if necessary
          out_buf.create_with_mask(size, cudf::mask_state::ALL_VALID, _stream, _mr);
        }
      }
    }
  }
}

std::vector<size_t> reader::impl::calculate_page_string_offsets()
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  auto page_keys = make_page_key_iterator(subpass.pages);

  std::vector<size_t> col_sizes(_input_columns.size(), 0L);
  rmm::device_uvector<size_t> d_col_sizes(col_sizes.size(), _stream);

  // use page_index to fetch page string sizes in the proper order
  auto val_iter = thrust::make_transform_iterator(subpass.pages.device_begin(),
                                                  page_to_string_size{pass.chunks.d_begin()});

  // do scan by key to calculate string offsets for each page
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(_stream),
                                page_keys,
                                page_keys + subpass.pages.size(),
                                val_iter,
                                page_offset_output_iter{subpass.pages.device_ptr()});

  // now sum up page sizes
  rmm::device_uvector<int> reduce_keys(col_sizes.size(), _stream);
  thrust::reduce_by_key(rmm::exec_policy_nosync(_stream),
                        page_keys,
                        page_keys + subpass.pages.size(),
                        val_iter,
                        reduce_keys.begin(),
                        d_col_sizes.begin());

  cudaMemcpyAsync(col_sizes.data(),
                  d_col_sizes.data(),
                  sizeof(size_t) * col_sizes.size(),
                  cudaMemcpyDeviceToHost,
                  _stream);
  _stream.synchronize();

  return col_sizes;
}

}  // namespace cudf::io::parquet::detail
