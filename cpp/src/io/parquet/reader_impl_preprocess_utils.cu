/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "error.hpp"
#include "io/comp/common.hpp"
#include "reader_impl_preprocess_utils.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <bitset>
#include <iostream>

namespace cudf::io::parquet::detail {

#if defined(PREPROCESS_DEBUG)
void print_pages(cudf::detail::hostdevice_span<PageInfo> pages, rmm::cuda_stream_view stream)
{
  pages.device_to_host(stream);
  auto idx = 0;
  std::for_each(pages.host_begin(), pages.host_end(), [&](PageInfo const& p) {
    if (p.flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY) { std::cout << "Dict"; }
    std::cout << "P(" << idx++ << ", s:" << p.src_col_schema << "): chunk_row(" << p.chunk_row
              << "), num_rows(" << p.num_rows << "), skipped_values(" << p.skipped_values
              << "), skipped_leaf_values(" << p.skipped_leaf_values << "), str_bytes("
              << p.str_bytes << ")\n";
  });
}
#endif  // PREPROCESS_DEBUG

void generate_depth_remappings(
  std::map<std::pair<int, int>, std::pair<std::vector<int>, std::vector<int>>>& remap,
  int const src_col_schema,
  int const mapped_src_col_schema,
  int const src_file_idx,
  aggregate_reader_metadata const& md)
{
  // already generated for this level
  if (remap.find({src_col_schema, src_file_idx}) != remap.end()) { return; }
  auto const& schema   = md.get_schema(mapped_src_col_schema, src_file_idx);
  auto const max_depth = md.get_output_nesting_depth(src_col_schema);

  CUDF_EXPECTS(remap.find({src_col_schema, src_file_idx}) == remap.end(),
               "Attempting to remap a schema more than once");
  auto inserted =
    remap.insert(std::pair<std::pair<int, int>, std::pair<std::vector<int>, std::vector<int>>>{
      {src_col_schema, src_file_idx}, {}});
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
      int schema_idx = mapped_src_col_schema;
      while (schema_idx > 0) {
        auto& cur_schema = md.get_schema(schema_idx, src_file_idx);
        if (cur_schema.max_repetition_level == r) {
          // if this is a repeated field, map it one level deeper
          shallowest = cur_schema.is_stub() ? cur_depth + 1 : cur_depth;
        }
        // if it's one-level encoding list
        else if (cur_schema.is_one_level_list(md.get_schema(cur_schema.parent_idx, src_file_idx))) {
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
      int schema_idx = mapped_src_col_schema;
      int r1         = 0;
      while (schema_idx > 0) {
        SchemaElement cur_schema = md.get_schema(schema_idx, src_file_idx);
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
      schema_idx = mapped_src_col_schema;
      int depth  = max_depth - 1;
      while (schema_idx > 0) {
        SchemaElement cur_schema = md.get_schema(schema_idx, src_file_idx);
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

[[nodiscard]] std::future<void> read_column_chunks_async(
  std::vector<std::unique_ptr<datasource>> const& sources,
  cudf::host_span<rmm::device_buffer> page_data,
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
    size_t const io_offset = column_chunk_offsets[chunk];
    size_t io_size         = chunks[chunk].compressed_size;
    size_t next_chunk      = chunk + 1;
    while (next_chunk < end_chunk) {
      size_t const next_offset = column_chunk_offsets[next_chunk];
      if (next_offset != io_offset + io_size ||
          chunk_source_map[chunk] != chunk_source_map[next_chunk]) {
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
        page_data[chunk] = rmm::device_buffer(
          cudf::util::round_up_safe(io_size, cudf::io::detail::BUFFER_PADDING_MULTIPLE), stream);
        auto fut_read_size = source->device_read_async(
          io_offset, io_size, static_cast<uint8_t*>(page_data[chunk].data()), stream);
        read_tasks.emplace_back(std::move(fut_read_size));
      } else {
        auto const read_buffer = source->host_read(io_offset, io_size);
        // Buffer needs to be padded.
        // Required by `gpuDecodePageData`.
        page_data[chunk] = rmm::device_buffer(
          cudf::util::round_up_safe(read_buffer->size(), cudf::io::detail::BUFFER_PADDING_MULTIPLE),
          stream);
        CUDF_CUDA_TRY(cudaMemcpyAsync(page_data[chunk].data(),
                                      read_buffer->data(),
                                      read_buffer->size(),
                                      cudaMemcpyDefault,
                                      stream));
      }
      auto d_compdata = static_cast<uint8_t const*>(page_data[chunk].data());
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
      task.get();
    }
  };
  return std::async(std::launch::deferred, sync_fn, std::move(read_tasks));
}

[[nodiscard]] size_t count_page_headers(cudf::detail::hostdevice_vector<ColumnChunkDesc>& chunks,
                                        rmm::cuda_stream_view stream)
{
  size_t total_pages = 0;

  kernel_error error_code(stream);
  chunks.host_to_device_async(stream);
  decode_page_headers(chunks.device_ptr(), nullptr, chunks.size(), error_code.data(), stream);
  chunks.device_to_host(stream);

  // It's required to ignore unsupported encodings in this function
  // so that we can actually compile a list of all the unsupported encodings found
  // in the pages. That cannot be done here since we do not have the pages vector here.
  // see https://github.com/rapidsai/cudf/pull/14453#pullrequestreview-1778346688
  if (auto const error = error_code.value_sync(stream);
      error != 0 and error != static_cast<uint32_t>(decode_error::UNSUPPORTED_ENCODING)) {
    CUDF_FAIL("Parquet header parsing failed with code(s) while counting page headers " +
              kernel_error::to_string(error));
  }

  for (auto& chunk : chunks) {
    total_pages += chunk.num_data_pages + chunk.num_dict_pages;
  }

  return total_pages;
}

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

void fill_in_page_info(host_span<ColumnChunkDesc> chunks,
                       device_span<PageInfo> pages,
                       rmm::cuda_stream_view stream)
{
  auto const num_pages = pages.size();
  auto page_indexes    = cudf::detail::make_host_vector<page_index_info>(num_pages, stream);

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
    page_indexes, stream, cudf::get_current_device_resource_ref());

  auto iter = thrust::make_counting_iterator<size_type>(0);
  thrust::for_each(
    rmm::exec_policy_nosync(stream), iter, iter + num_pages, copy_page_info{d_page_indexes, pages});
}

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

[[nodiscard]] std::string list_unsupported_encodings(device_span<PageInfo const> pages,
                                                     rmm::cuda_stream_view stream)
{
  auto const to_mask = cuda::proclaim_return_type<uint32_t>([] __device__(auto const& page) {
    return is_supported_encoding(page.encoding) ? uint32_t{0} : encoding_to_mask(page.encoding);
  });
  uint32_t const unsupported = thrust::transform_reduce(rmm::exec_policy(stream),
                                                        pages.begin(),
                                                        pages.end(),
                                                        to_mask,
                                                        uint32_t{0},
                                                        cuda::std::bit_or<uint32_t>());
  return encoding_bitmask_to_str(unsupported);
}

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
                             cuda::std::less<int>());
  auto pass_pages = cudf::detail::hostdevice_vector<PageInfo>(unsorted_pages.size(), stream);
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

void decode_page_headers(pass_intermediate_data& pass,
                         device_span<PageInfo> unsorted_pages,
                         bool has_page_index,
                         rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  auto iter = thrust::counting_iterator<size_t>(0);
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
    size_t{0},
    cuda::std::plus<size_t>{});
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
  decode_page_headers(pass.chunks.d_begin(),
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
                                            cuda::maximum<int>());
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

}  // namespace cudf::io::parquet::detail
