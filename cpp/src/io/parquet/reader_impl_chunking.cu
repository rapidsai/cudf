/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include "reader_impl_chunking.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <io/comp/nvcomp_adapter.hpp>

#include <io/utilities/time_utils.cuh>
#include <io/utilities/config_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <numeric>

namespace cudf::io::parquet::detail {

namespace {

struct cumulative_page_info {
  size_t row_count;   // cumulative row count
  size_t size_bytes;  // cumulative size in bytes
  int key;            // schema index
};

struct split_info {
  row_range rows;
  int64_t split_pos;
};

#if defined(CHUNKING_DEBUG)
void print_cumulative_page_info(cudf::detail::hostdevice_vector<PageInfo>& pages,
                                rmm::device_uvector<cumulative_page_info> const& c_info,
                                rmm::cuda_stream_view stream)
{
  pages.device_to_host_sync(stream);

  printf("------------\nCumulative sizes by page\n");

  std::vector<int> schemas(pages.size());
  std::vector<cumulative_page_info> h_cinfo(pages.size());
  CUDF_CUDA_TRY(cudaMemcpy(
    h_cinfo.data(), c_info.data(), sizeof(cumulative_page_info) * pages.size(), cudaMemcpyDefault));
  auto schema_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](size_type i) { return pages[i].src_col_schema; });
  thrust::copy(thrust::seq, schema_iter, schema_iter + pages.size(), schemas.begin());
  auto last = thrust::unique(thrust::seq, schemas.begin(), schemas.end());
  schemas.resize(last - schemas.begin());
  printf("Num schemas: %lu\n", schemas.size());

  for (size_t idx = 0; idx < schemas.size(); idx++) {
    printf("Schema %d\n", schemas[idx]);
    for (size_t pidx = 0; pidx < pages.size(); pidx++) {
      auto const& page = pages[pidx];
      if (page.flags & PAGEINFO_FLAGS_DICTIONARY || page.src_col_schema != schemas[idx]) {
        continue;
      }
      printf("\tP: {%lu, %lu}\n", h_cinfo[pidx].row_count, h_cinfo[pidx].size_bytes);
    }
  }
}

void print_cumulative_row_info(host_span<cumulative_page_info const> sizes,
                               std::string const& label,
                               std::optional<std::vector<split_info>> splits = std::nullopt)
{
  if (splits.has_value()) {
    printf("------------\nSplits (skip_rows, num_rows)\n");
    for (size_t idx = 0; idx < splits->size(); idx++) {
      printf("{%lu, %lu}\n", splits.value()[idx].rows.skip_rows, splits.value()[idx].rows.num_rows);
    }
  }

  printf("------------\nCumulative sizes %s (row_count, size_bytes, page_key)\n", label.c_str());
  for (size_t idx = 0; idx < sizes.size(); idx++) {
    printf("{%lu, %lu, %d}", sizes[idx].row_count, sizes[idx].size_bytes, sizes[idx].key);
    if (splits.has_value()) {
      // if we have a split at this row count and this is the last instance of this row count
      auto start = thrust::make_transform_iterator(
        splits->begin(), [](split_info const& i) { return i.rows.skip_rows; });
      auto end               = start + splits->size();
      auto split             = std::find(start, end, sizes[idx].row_count);
      auto const split_index = [&]() -> int {
        if (split != end &&
            ((idx == sizes.size() - 1) || (sizes[idx + 1].row_count > sizes[idx].row_count))) {
          return static_cast<int>(std::distance(start, split));
        }
        return idx == 0 ? 0 : -1;
      }();
      if (split_index >= 0) {
        printf(" <-- split {%lu, %lu, %lu}",
               splits.value()[split_index].rows.skip_rows,
               splits.value()[split_index].rows.num_rows,
               splits.value()[split_index].split_pos);
      }
    }
    printf("\n");
  }
}
#endif  // CHUNKING_DEBUG


/**
 * @brief Functor which reduces two cumulative_page_info structs of the same key.
 */
struct cumulative_page_sum {
  cumulative_page_info operator()
    __device__(cumulative_page_info const& a, cumulative_page_info const& b) const
  {
    return cumulative_page_info{a.row_count + b.row_count, a.size_bytes + b.size_bytes, a.key};
  }
};

/**
 * @brief Functor which computes the total data size for a given type of cudf column.
 *
 * In the case of strings, the return size does not include the chars themselves. That
 * information is tracked separately (see PageInfo::str_bytes).
 */
struct row_size_functor {
  __device__ size_t validity_size(size_t num_rows, bool nullable)
  {
    return nullable ? (cudf::util::div_rounding_up_safe(num_rows, size_t{32}) * 4) : 0;
  }

  template <typename T>
  __device__ size_t operator()(size_t num_rows, bool nullable)
  {
    auto const element_size = sizeof(device_storage_type_t<T>);
    return (element_size * num_rows) + validity_size(num_rows, nullable);
  }
};

template <>
__device__ size_t row_size_functor::operator()<list_view>(size_t num_rows, bool nullable)
{
  auto const offset_size = sizeof(size_type);
  // NOTE: Adding the + 1 offset here isn't strictly correct.  There will only be 1 extra offset
  // for the entire column, whereas this is adding an extra offset per page.  So we will get a
  // small over-estimate of the real size of the order :  # of pages * 4 bytes. It seems better
  // to overestimate size somewhat than to underestimate it and potentially generate chunks
  // that are too large.
  return (offset_size * (num_rows + 1)) + validity_size(num_rows, nullable);
}

template <>
__device__ size_t row_size_functor::operator()<struct_view>(size_t num_rows, bool nullable)
{
  return validity_size(num_rows, nullable);
}

template <>
__device__ size_t row_size_functor::operator()<string_view>(size_t num_rows, bool nullable)
{
  // only returns the size of offsets and validity. the size of the actual string chars
  // is tracked separately.
  auto const offset_size = sizeof(size_type);
  // see note about offsets in the list_view template.
  return (offset_size * (num_rows + 1)) + validity_size(num_rows, nullable);
}

/**
 * @brief Functor which computes the total output cudf data size for all of
 * the data in this page.
 *
 * Sums across all nesting levels.
 */
struct get_cumulative_page_info {
  __device__ cumulative_page_info operator()(PageInfo const& page)
  {    
    if (page.flags & PAGEINFO_FLAGS_DICTIONARY) {
      return cumulative_page_info{0, 0, page.src_col_schema};
    }

    // total nested size, not counting string data
    auto iter =
      cudf::detail::make_counting_transform_iterator(0, [page] __device__(size_type i) {
        auto const& pni = page.nesting[i];
        return cudf::type_dispatcher(
          data_type{pni.type}, row_size_functor{}, pni.size, pni.nullable);
      });

    size_t const row_count = static_cast<size_t>(page.nesting[0].size);
    return {
      row_count,
      thrust::reduce(thrust::seq, iter, iter + page.num_output_nesting_levels) + page.str_bytes,
      page.src_col_schema};
  }
};

/**
 * @brief Functor which computes the (uncompressed) size of a page.
 */
struct get_page_size {
  device_span<const ColumnChunkDesc> chunks;

  __device__ cumulative_page_info operator()(PageInfo const& page)
  {
    if (page.flags & PAGEINFO_FLAGS_DICTIONARY) {
      return cumulative_page_info{0, 0, page.src_col_schema};
    }
    // TODO: this is not accurate for lists. it might make sense to make a guess
    // based on total-rowgroup-size / # of rows in the rowgroup for an average of
    // rows-per-byte.     
    size_t const row_count = page.num_rows;
    return {
      row_count,
      static_cast<size_t>(page.uncompressed_page_size),
      page.src_col_schema};
  }
};

/**
 * @brief Functor which computes the effective size of all input columns by page.
 *
 * For a given row, we want to find the cost of all pages for all columns involved
 * in loading up to that row.  The complication here is that not all pages are the
 * same size between columns. Example:
 *
 *              page row counts
 * Column A:    0 <----> 100 <----> 200
 * Column B:    0 <---------------> 200 <--------> 400
                          |
 * if we decide to split at row 100, we don't really know the actual amount of bytes in column B
 * at that point.  So we have to proceed as if we are taking the bytes from all 200 rows of that
 * page. Essentially, a conservative over-estimate of the real size.
 */
struct page_total_size {
  cumulative_page_info const* c_info;
  size_type const* key_offsets;
  size_t num_keys;

  __device__ cumulative_page_info operator()(cumulative_page_info const& i)
  {
    // sum sizes for each input column at this row
    size_t sum = 0;
    for (int idx = 0; idx < num_keys; idx++) {
      auto const start = key_offsets[idx];
      auto const end   = key_offsets[idx + 1];
      auto iter        = cudf::detail::make_counting_transform_iterator(
        0, [&] __device__(size_type i) { return c_info[i].row_count; });
      auto const page_index =
        thrust::lower_bound(thrust::seq, iter + start, iter + end, i.row_count) - iter;
      sum += c_info[page_index].size_bytes;
    }
    return {i.row_count, sum, i.key};
  }
};

int64_t find_next_split(int64_t cur_pos,
                        size_t cur_row_count,
                        std::vector<cumulative_page_info> const& sizes,
                        size_t chunk_read_limit)
{
  size_t cur_cumulative_size = cur_pos == 0 ? 0 : sizes[cur_pos-1].size_bytes;

  auto start = thrust::make_transform_iterator(sizes.begin(), [&](cumulative_page_info const& i) {
    return i.size_bytes - cur_cumulative_size;
  });
  auto end   = start + sizes.size();
  
  int64_t split_pos =
    thrust::lower_bound(thrust::seq, start + cur_pos, end, chunk_read_limit) - start;

  // if we're past the end, or if the returned bucket is > than the chunk_read_limit, move back
  // one.
  if (static_cast<size_t>(split_pos) >= sizes.size() ||
      (sizes[split_pos].size_bytes - cur_cumulative_size > chunk_read_limit)) {
    split_pos--;
  }

  // best-try. if we can't find something that'll fit, we have to go bigger. we're doing this in
  // a loop because all of the cumulative sizes for all the pages are sorted into one big list.
  // so if we had two columns, both of which had an entry {1000, 10000}, that entry would be in
  // the list twice. so we have to iterate until we skip past all of them.  The idea is that we
  // either do this, or we have to call unique() on the input first.
  while (split_pos < (static_cast<int64_t>(sizes.size()) - 1) &&
          (split_pos < 0 || sizes[split_pos].row_count == cur_row_count)) {
    split_pos++;
  }

  return split_pos;
}

/**
 * @brief Given a vector of cumulative {row_count, byte_size} pairs and a chunk read
 * limit, determine the set of splits.
 *
 * @param sizes Vector of cumulative {row_count, byte_size} pairs
 * @param chunk_read_limit Limit on total number of bytes to be returned per read, for all columns
 */
std::vector<split_info> find_splits(std::vector<cumulative_page_info> const& sizes,
                                   size_t chunk_read_limit)
{
  // now we have an array of {row_count, real output bytes}. just walk through it and generate
  // splits.
  // TODO: come up with a clever way to do this entirely in parallel. For now, as long as batch
  // sizes are reasonably large, this shouldn't iterate too many times
  std::vector<split_info> splits;
  {
    size_t cur_pos             = 0;
    size_t cur_row_count       = 0;
    auto const num_rows = sizes.back().row_count;
    while (cur_row_count < num_rows) {
      auto const split_pos = find_next_split(cur_pos, cur_row_count, sizes, chunk_read_limit);
      
      auto const start_row = cur_row_count;
      cur_row_count        = sizes[split_pos].row_count;
      splits.push_back(split_info{row_range{start_row, cur_row_count - start_row}, static_cast<int64_t>(cur_pos == 0 ? 0 : cur_pos + 1)});
      cur_pos             = split_pos;
    }
  }
  // print_cumulative_row_info(sizes, "adjusted", splits);

  return splits;
}

/**
 * @brief Converts cuDF units to Parquet units.
 *
 * @return A tuple of Parquet type width, Parquet clock rate and Parquet decimal type.
 */
[[nodiscard]] std::tuple<int32_t, int32_t, int8_t> conversion_info(type_id column_type_id,
                                                                   type_id timestamp_type_id,
                                                                   Type physical,
                                                                   int8_t converted,
                                                                   int32_t length)
{
  int32_t type_width = (physical == FIXED_LEN_BYTE_ARRAY) ? length : 0;
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
  if (converted_type == DECIMAL && column_type_id != type_id::FLOAT64 &&
      not cudf::is_fixed_point(data_type{column_type_id})) {
    converted_type = UNKNOWN;  // Not converting to float64 or decimal
  }
  return std::make_tuple(type_width, clock_rate, converted_type);
}

/**
 * @brief Return the required number of bits to store a value.
 */
template <typename T = uint8_t>
[[nodiscard]] T required_bits(uint32_t max_level)
{
  return static_cast<T>(CompactProtocolReader::NumRequiredBits(max_level));
}

struct row_count_compare {
  __device__ bool operator()(cumulative_page_info const& a, cumulative_page_info const& b)
  {
    return a.row_count < b.row_count;
  }
};

std::pair<size_t, size_t> get_row_group_size(RowGroup const& rg)
{
  auto compressed_size_iter = thrust::make_transform_iterator(rg.columns.begin(), [](ColumnChunk const& c){
    return c.meta_data.total_compressed_size;
  });

  // the trick is that total temp space needed is tricky to know
  auto const compressed_size = std::reduce(compressed_size_iter, compressed_size_iter + rg.columns.size());
  auto const total_size = compressed_size + rg.total_byte_size;
  return {compressed_size, total_size};
}

std::pair<rmm::device_uvector<cumulative_page_info>, rmm::device_uvector<int32_t>>
adjust_cumulative_sizes(rmm::device_uvector<cumulative_page_info> const& c_info,
                        cudf::detail::hostdevice_vector<PageInfo> const& pages,
                        rmm::cuda_stream_view stream)
{
  // sort by row count
  rmm::device_uvector<cumulative_page_info> c_info_sorted{c_info, stream};  
  thrust::sort(
    rmm::exec_policy(stream), c_info_sorted.begin(), c_info_sorted.end(), row_count_compare{});

  // page keys grouped by split.
  rmm::device_uvector<int32_t> page_keys_by_split{c_info.size(), stream};
  thrust::transform(rmm::exec_policy(stream), c_info_sorted.begin(), c_info_sorted.end(), page_keys_by_split.begin(), [] __device__ (cumulative_page_info const& c){
    return c.key;
  });

  std::vector<cumulative_page_info> h_c_info_sorted(c_info_sorted.size());
  CUDF_CUDA_TRY(cudaMemcpy(h_c_info_sorted.data(),
                            c_info_sorted.data(),
                            sizeof(cumulative_page_info) * c_info_sorted.size(),
                            cudaMemcpyDefault));
  // print_cumulative_row_info(h_c_info_sorted, "raw");

  // generate key offsets (offsets to the start of each partition of keys). worst case is 1 page per
  // key
  rmm::device_uvector<size_type> key_offsets(pages.size() + 1, stream);
  auto page_keys = make_page_key_iterator(pages);
  auto const key_offsets_end = thrust::reduce_by_key(rmm::exec_policy(stream),
                                                     page_keys,
                                                     page_keys + pages.size(),
                                                     thrust::make_constant_iterator(1),
                                                     thrust::make_discard_iterator(),
                                                     key_offsets.begin()).second;
  size_t const num_unique_keys = key_offsets_end - key_offsets.begin();
  thrust::exclusive_scan(
    rmm::exec_policy(stream), key_offsets.begin(), key_offsets.end(), key_offsets.begin());

  // adjust the cumulative info such that for each row count, the size includes any pages that span
  // that row count. this is so that if we have this case:
  //              page row counts
  // Column A:    0 <----> 100 <----> 200
  // Column B:    0 <---------------> 200 <--------> 400
  //                        |
  // if we decide to split at row 100, we don't really know the actual amount of bytes in column B
  // at that point.  So we have to proceed as if we are taking the bytes from all 200 rows of that
  // page.
  //
  rmm::device_uvector<cumulative_page_info> aggregated_info(c_info.size(), stream);
  thrust::transform(rmm::exec_policy(stream),
                    c_info_sorted.begin(),
                    c_info_sorted.end(),
                    aggregated_info.begin(),
                    page_total_size{c_info.data(), key_offsets.data(), num_unique_keys});
  return {std::move(aggregated_info), std::move(page_keys_by_split)};
}

struct page_span {
  size_t start, end;
};
std::pair<std::vector<page_span>, size_t>
compute_next_subpass(rmm::device_uvector<cumulative_page_info> const& c_info,
                     cudf::detail::hostdevice_vector<PageInfo> const& pages,
                     cudf::detail::hostdevice_vector<size_type> const& page_offsets,
                     size_t min_row,
                     size_t size_limit,
                     size_t num_columns,
                     rmm::cuda_stream_view stream)
{
  auto [aggregated_info, page_keys_by_split] = adjust_cumulative_sizes(c_info, pages, stream);

  // bring back to the cpu
  std::vector<cumulative_page_info> h_aggregated_info(aggregated_info.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_aggregated_info.data(),
                                aggregated_info.data(),
                                sizeof(cumulative_page_info) * c_info.size(),
                                cudaMemcpyDefault,
                                stream.value()));
  stream.synchronize();

  // print_cumulative_row_info(h_aggregated_info, "adjusted");

  // first, find the min row
  auto start = thrust::make_transform_iterator(h_aggregated_info.begin(), [&](cumulative_page_info const& i){
    return i.row_count;
  });
  auto const start_index = thrust::upper_bound(thrust::host, start, start + h_aggregated_info.size(), min_row) - start;    

  // find the next split
  auto const end_index = find_next_split(start_index,
                                         min_row,
                                         // 0,
                                         h_aggregated_info,
                                         size_limit) + 1; // the split index returned is inclusive

  // get the number of pages for each column/schema
  auto get_page_counts = [num_columns, stream](rmm::device_uvector<cumulative_page_info> const& aggregated_info, int start_index, int end_index){
    std::vector<size_t> h_page_counts(num_columns);

    auto const num_pages = end_index - start_index;
    if(num_pages == 0){
      std::fill(h_page_counts.begin(), h_page_counts.end(), 0);
      return h_page_counts;
    }

    rmm::device_uvector<int32_t> page_keys(num_pages, stream);
    thrust::transform(rmm::exec_policy(stream), 
                      aggregated_info.begin() + start_index, 
                      aggregated_info.begin() + end_index,
                      page_keys.begin(),
                      [] __device__ (cumulative_page_info const& i){
                        return i.key;
                      });    
    thrust::sort(rmm::exec_policy(stream), page_keys.begin(), page_keys.end());
    rmm::device_uvector<size_t> page_counts(num_pages, stream);
    auto page_counts_end = thrust::reduce_by_key(rmm::exec_policy(stream), 
                                                 page_keys.begin(), 
                                                 page_keys.end(), 
                                                 thrust::make_constant_iterator(1), 
                                                 thrust::make_discard_iterator(),
                                                 page_counts.begin()).second;
    auto const num_page_counts = page_counts_end - page_counts.begin();
    CUDF_EXPECTS(static_cast<size_t>(num_page_counts) == num_columns, "Encountered a mismatch in column/schema counts while computing subpass split");
    
    cudaMemcpyAsync(h_page_counts.data(), page_counts.data(), sizeof(size_t) * num_columns, cudaMemcpyDeviceToHost);
    stream.synchronize();
    return h_page_counts;
  };

  // get count of pages before this split and in this split.
  auto last_counts = get_page_counts(aggregated_info, 0, start_index);
  auto this_counts = get_page_counts(aggregated_info, start_index, end_index);

  // convert to page spans
  std::vector<page_span> out(num_columns);
  size_t total_pages = 0;
  for(size_t c_idx=0; c_idx<num_columns; c_idx++){
    // add page_offsets to get proper indices into the pages array
    out[c_idx].start = (last_counts[c_idx]) + page_offsets[c_idx];
    out[c_idx].end = (last_counts[c_idx] + this_counts[c_idx]) + page_offsets[c_idx];
    total_pages += this_counts[c_idx];
  }

  return {out, total_pages};
}


std::pair<std::vector<split_info>, rmm::device_uvector<int32_t>>
compute_page_splits_by_row(rmm::device_uvector<cumulative_page_info> const& c_info,
                           cudf::detail::hostdevice_vector<PageInfo> const& pages,
                           size_t size_limit,
                           rmm::cuda_stream_view stream)
{
  auto [aggregated_info, page_keys_by_split] = adjust_cumulative_sizes(c_info, pages, stream);

  // bring back to the cpu
  std::vector<cumulative_page_info> h_aggregated_info(aggregated_info.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_aggregated_info.data(),
                                aggregated_info.data(),
                                sizeof(cumulative_page_info) * c_info.size(),
                                cudaMemcpyDefault,
                                stream.value()));
  stream.synchronize();

  // generate the actual splits
  return {find_splits(h_aggregated_info, size_limit), std::move(page_keys_by_split)};
}

/**
 * @brief Decompresses the page data, at page granularity.
 * 
 * This function handles the case where `pages` is only a subset of all available
 * pages in `chunks`.
 *
 * @param chunks List of column chunk descriptors
 * @param pages List of page information
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Device buffer to decompressed page data
 */
[[nodiscard]] rmm::device_buffer decompress_page_data(
  cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
  cudf::detail::hostdevice_vector<PageInfo>& pages,
  rmm::cuda_stream_view stream)
{
  auto for_each_codec_page = [&](Compression codec, std::function<void(size_t)> const& f) {
    for(size_t p = 0; p<pages.size(); p++){
      if(chunks[pages[p].chunk_idx].codec == codec){
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
    Compression compression_type  = UNCOMPRESSED;
    size_t num_pages              = 0;
    int32_t max_decompressed_size = 0;
    size_t total_decomp_size      = 0;
  };

  std::array codecs{codec_stats{GZIP}, codec_stats{SNAPPY}, codec_stats{BROTLI}, codec_stats{ZSTD}};

  auto is_codec_supported = [&codecs](int8_t codec) {
    if (codec == UNCOMPRESSED) return true;
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
    if (codec.compression_type == BROTLI && codec.num_pages > 0) {
      debrotli_scratch.resize(get_gpu_debrotli_scratch_size(codec.num_pages), stream);
    }
  }

  // Dispatch batches of pages to decompress for each codec.
  // Buffer needs to be padded, required by `gpuDecodePageData`.
  rmm::device_buffer decomp_pages(
    cudf::util::round_up_safe(total_decomp_size, BUFFER_PADDING_MULTIPLE), stream);

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
  for (auto const& codec : codecs) {
    if (codec.num_pages == 0) { continue; }

    for_each_codec_page(codec.compression_type, [&](size_t page_idx) {
      auto const dst_base = static_cast<uint8_t*>(decomp_pages.data()) + decomp_offset;
      auto& page          = pages[page_idx];
      // offset will only be non-zero for V2 pages
      auto const offset =
        page.lvl_bytes[level_type::DEFINITION] + page.lvl_bytes[level_type::REPETITION];
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
    auto const d_comp_in = cudf::detail::make_device_uvector_async(
      comp_in_view, stream, rmm::mr::get_current_device_resource());
    host_span<device_span<uint8_t> const> comp_out_view(comp_out.data() + start_pos,
                                                        codec.num_pages);
    auto const d_comp_out = cudf::detail::make_device_uvector_async(
      comp_out_view, stream, rmm::mr::get_current_device_resource());
    device_span<compression_result> d_comp_res_view(comp_res.data() + start_pos, codec.num_pages);

    switch (codec.compression_type) {
      case GZIP:
        gpuinflate(d_comp_in, d_comp_out, d_comp_res_view, gzip_header_included::YES, stream);
        break;
      case SNAPPY:
        if (cudf::io::detail::nvcomp_integration::is_stable_enabled()) {
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
      case ZSTD:
        nvcomp::batched_decompress(nvcomp::compression_type::ZSTD,
                                   d_comp_in,
                                   d_comp_out,
                                   d_comp_res_view,
                                   codec.max_decompressed_size,
                                   codec.total_decomp_size,
                                   stream);
        break;
      case BROTLI:
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
    auto const d_copy_in = cudf::detail::make_device_uvector_async(
      copy_in, stream, rmm::mr::get_current_device_resource());
    auto const d_copy_out = cudf::detail::make_device_uvector_async(
      copy_out, stream, rmm::mr::get_current_device_resource());

    gpu_copy_uncompressed_blocks(d_copy_in, d_copy_out, stream);
    stream.synchronize();
  }

  // Update the page information in device memory with the updated value of
  // page_data; it now points to the uncompressed data buffer
  pages.host_to_device_async(stream);

  return decomp_pages;
}

struct flat_column_num_rows {
  ColumnChunkDesc const* chunks;

  __device__ size_type operator()(PageInfo const& page) const
  {
    // ignore dictionary pages and pages belonging to any column containing repetition (lists)
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) ||
        (chunks[page.chunk_idx].max_level[level_type::REPETITION] > 0)) {
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
void detect_malformed_pages(cudf::detail::hostdevice_vector<PageInfo> const& pages,
                            cudf::detail::hostdevice_vector<ColumnChunkDesc> const& chunks,
                            std::optional<size_t> expected_row_count,
                            rmm::cuda_stream_view stream)
{
  // sum row counts for all non-dictionary, non-list columns. other columns will be indicated as 0
  rmm::device_uvector<size_type> row_counts(pages.size(),
                                            stream);  // worst case:  num keys == num pages
  auto const size_iter = thrust::make_transform_iterator(pages.d_begin(), flat_column_num_rows{chunks.device_ptr()});
  auto const row_counts_begin = row_counts.begin();
  auto page_keys = make_page_key_iterator(pages);
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
    size_t const found_row_count = static_cast<size_t>(compacted_row_counts.element(0, stream));

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

}  // anonymous namespace

void reader::impl::create_global_chunk_info()
{
  auto const num_rows         = _file_itm_data.global_num_rows;
  auto const& row_groups_info = _file_itm_data.row_groups;
  auto& chunks                = _file_itm_data.chunks;

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_chunks        = row_groups_info.size() * num_input_columns;

  // Initialize column chunk information
  auto remaining_rows = num_rows;
  for (auto const& rg : row_groups_info) {
    auto const& row_group      = _metadata->get_row_group(rg.index, rg.source_index);
    auto const row_group_start = rg.start_row;
    auto const row_group_rows  = std::min<int>(remaining_rows, row_group.num_rows);

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

      chunks.push_back(ColumnChunkDesc(col_meta.total_compressed_size,
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
                                       schema.decimal_precision,
                                       clock_rate,
                                       i,
                                       col.schema_idx));
    }

    remaining_rows -= row_group_rows;
  }
}

void reader::impl::compute_input_passes()
{
  // at this point, row_groups has already been filtered down to just the row groups we need to
  // handle optional skip_rows/num_rows parameters.
  auto const& row_groups_info = _file_itm_data.row_groups;

  // if the user hasn't specified an input size limit, read everything in a single pass.
  if (_input_pass_read_limit == 0) {
    _file_itm_data.input_pass_row_group_offsets.push_back(0);
    _file_itm_data.input_pass_row_group_offsets.push_back(row_groups_info.size());
    return;
  }

  // generate passes. make sure to account for the case where a single row group doesn't fit within
  //
  std::size_t const read_limit =
    _input_pass_read_limit > 0 ? _input_pass_read_limit : std::numeric_limits<std::size_t>::max();
  std::size_t cur_pass_byte_size = 0;
  std::size_t cur_rg_start       = 0;
  std::size_t cur_row_count      = 0;
  _file_itm_data.input_pass_row_group_offsets.push_back(0);
  _file_itm_data.input_pass_row_count.push_back(0);

  for (size_t cur_rg_index = 0; cur_rg_index < row_groups_info.size(); cur_rg_index++) {
    auto const& rgi       = row_groups_info[cur_rg_index];
    auto const& row_group = _metadata->get_row_group(rgi.index, rgi.source_index);

    // total compressed size and total size (compressed + uncompressed) for 
    auto const [compressed_rg_size, _/*compressed + uncompressed*/] = get_row_group_size(row_group);

    // can we add this row group
    if (cur_pass_byte_size + compressed_rg_size >= read_limit) {
      // A single row group (the current one) is larger than the read limit:
      // We always need to include at least one row group, so end the pass at the end of the current
      // row group
      if (cur_rg_start == cur_rg_index) {
        _file_itm_data.input_pass_row_group_offsets.push_back(cur_rg_index + 1);
        _file_itm_data.input_pass_row_count.push_back(cur_row_count + row_group.num_rows);
        cur_rg_start       = cur_rg_index + 1;
        cur_pass_byte_size = 0;
      }
      // End the pass at the end of the previous row group
      else {
        _file_itm_data.input_pass_row_group_offsets.push_back(cur_rg_index);
        _file_itm_data.input_pass_row_count.push_back(cur_row_count);
        cur_rg_start       = cur_rg_index;
        cur_pass_byte_size = compressed_rg_size;
      }
    } else {
      cur_pass_byte_size += compressed_rg_size;
    }
    cur_row_count += row_group.num_rows;
  }

  // add the last pass if necessary
  if (_file_itm_data.input_pass_row_group_offsets.back() != row_groups_info.size()) {
    _file_itm_data.input_pass_row_group_offsets.push_back(row_groups_info.size());
    _file_itm_data.input_pass_row_count.push_back(cur_row_count);
  }
}

void reader::impl::compute_chunks_for_subpass()
{
  auto& pass = *_pass_itm_data;
  auto& subpass = *pass.subpass;
 
  // simple case : no chunk size, no splits
  if (_output_chunk_read_limit <= 0) {
    subpass.output_chunk_read_info.push_back({subpass.skip_rows, subpass.num_rows});
    return;
  }
  
  // generate cumulative row counts and sizes
  rmm::device_uvector<cumulative_page_info> c_info(subpass.pages.size(), _stream);
  // convert PageInfo to cumulative_page_info
  auto page_input = thrust::make_transform_iterator(subpass.pages.d_begin(), get_cumulative_page_info{});
  auto page_keys = make_page_key_iterator(subpass.pages);
  thrust::inclusive_scan_by_key(rmm::exec_policy(_stream),
                                page_keys,
                                page_keys + subpass.pages.size(),
                                page_input,
                                c_info.begin(),
                                thrust::equal_to{},
                                cumulative_page_sum{});
  // print_cumulative_page_info(subpass.pages, c_info, _stream);
  
  // compute the splits
  auto [splits, _] = compute_page_splits_by_row(c_info, subpass.pages, _output_chunk_read_limit, _stream);
  subpass.output_chunk_read_info.reserve(splits.size());

  // apply skip_rows from the subpass
  std::transform(splits.begin(), splits.end(), std::back_inserter(subpass.output_chunk_read_info), [&subpass](split_info const &s){
    row_range r = s.rows;
    r.skip_rows += subpass.skip_rows;
    return r;
  });
}

void reader::impl::preprocess_next_pass()
{
  auto const num_passes = _file_itm_data.input_pass_row_group_offsets.size() - 1;

  // always create the pass struct, even if we end up with no work.
  // this will also cause the previous pass information to be deleted
  _pass_itm_data = std::make_unique<pass_intermediate_data>();

  if (_file_itm_data.global_num_rows > 0 && not _file_itm_data.row_groups.empty() &&
      not _input_columns.empty() && _file_itm_data._current_input_pass < num_passes) {

    auto& pass = *_pass_itm_data;

    // setup row groups to be loaded for this pass
    auto const row_group_start = _file_itm_data.input_pass_row_group_offsets[_file_itm_data._current_input_pass];
    auto const row_group_end   = _file_itm_data.input_pass_row_group_offsets[_file_itm_data._current_input_pass + 1];
    auto const num_row_groups  = row_group_end - row_group_start;
    pass.row_groups.resize(num_row_groups);
    std::copy(_file_itm_data.row_groups.begin() + row_group_start,
              _file_itm_data.row_groups.begin() + row_group_end,
              pass.row_groups.begin());

    auto const num_passes = _file_itm_data.input_pass_row_group_offsets.size() - 1;
    CUDF_EXPECTS(_file_itm_data._current_input_pass < num_passes, "Encountered an invalid read pass index");

    auto const chunks_per_rowgroup = _input_columns.size();
    auto const num_chunks          = chunks_per_rowgroup * num_row_groups;

    auto chunk_start = _file_itm_data.chunks.begin() + (row_group_start * chunks_per_rowgroup);
    auto chunk_end   = _file_itm_data.chunks.begin() + (row_group_end * chunks_per_rowgroup);

    pass.chunks = cudf::detail::hostdevice_vector<ColumnChunkDesc>(num_chunks, _stream);
    std::copy(chunk_start, chunk_end, pass.chunks.begin());

    // compute skip_rows / num_rows for this pass.
    if (num_passes == 1) {
      pass.skip_rows = _file_itm_data.global_skip_rows;
      pass.num_rows  = _file_itm_data.global_num_rows;
    } else {
      auto const global_start_row = _file_itm_data.global_skip_rows;
      auto const global_end_row   = global_start_row + _file_itm_data.global_num_rows;
      auto const start_row =
        std::max(_file_itm_data.input_pass_row_count[_file_itm_data._current_input_pass], global_start_row);
      auto const end_row =
        std::min(_file_itm_data.input_pass_row_count[_file_itm_data._current_input_pass + 1], global_end_row);

      // skip_rows is always global in the sense that it is relative to the first row of
      // everything we will be reading, regardless of what pass we are on.
      // num_rows is how many rows we are reading this pass.
      pass.skip_rows =
        global_start_row + _file_itm_data.input_pass_row_count[_file_itm_data._current_input_pass];
      pass.num_rows = end_row - start_row;
    }

    // load page information for the chunk. this retrieves the compressed bytes for all the
    // pages, and their headers (which we can access without decompressing)
    load_compressed_data();

    // detect malformed columns.
    // - we have seen some cases in the wild where we have a row group containing N
    //   rows, but the total number of rows in the pages for column X is != N. while it
    //   is possible to load this by just capping the number of rows read, we cannot tell
    //   which rows are invalid so we may be returning bad data. in addition, this mismatch
    //   confuses the chunked reader
    detect_malformed_pages(pass.pages,
                           pass.chunks,
                           pass.num_rows,
                           _stream);

    // since there is only ever 1 dictionary per chunk (the 0th path), do it at the 
    // pass level.
    build_string_dict_indices();

    // compute offsets to each group of input pages. this also gives us the number of unique
    // columns in the input
    // page_keys:   1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
    //
    // result:      0,          4,          8    
    rmm::device_uvector<size_type> page_counts(pass.pages.size() + 1, _stream);
    auto page_keys = make_page_key_iterator(pass.pages);
    auto const page_counts_end = thrust::reduce_by_key(rmm::exec_policy(_stream),
                                                        page_keys,
                                                        page_keys + pass.pages.size(),
                                                        thrust::make_constant_iterator(1),
                                                        thrust::make_discard_iterator(),
                                                        page_counts.begin()).second;
    auto const num_page_counts = page_counts_end - page_counts.begin();
    pass.page_offsets = cudf::detail::hostdevice_vector<size_type>(num_page_counts + 1, _stream);
    thrust::exclusive_scan(
      rmm::exec_policy(_stream), page_counts.begin(), page_counts.begin() + num_page_counts + 1, pass.page_offsets.d_begin());
    pass.page_offsets.device_to_host_async(_stream);

    pass.page_processed_counts = std::vector<size_type>(num_page_counts);
    std::fill(pass.page_processed_counts.begin(), pass.page_processed_counts.end(), 0);

    // compute subpasses for this pass using the page information we now have.
    // compute_subpasses();
    /*
    if (_output_chunk_read_limit == 0) {  // read the whole file at once
      CUDF_EXPECTS(_pass_itm_data->output_chunk_read_info.size() == 1,
                    "Reading the whole file should yield only one chunk.");
    }
    */

    _stream.synchronize();
  }
}

void reader::impl::handle_chunking(bool uses_custom_row_bounds)
{  
  // if this is our first time in here, setup the first pass.
  if(!_pass_itm_data){
    // preprocess the next pass
    preprocess_next_pass();
  }

  auto& pass = *_pass_itm_data;

  // if we already have a subpass in flight.
  if(pass.subpass != nullptr){
    // if it still has more chunks in flight, there's nothing more to do
    if(pass.subpass->current_output_chunk < pass.subpass->output_chunk_read_info.size()){
      return;
    }    

    // release the old subpass (will free memory)
    pass.subpass.reset();

    // otherwise we are done with the pass entirely
    if(pass.processed_rows == pass.num_rows){
      // release the old pass
      _pass_itm_data.reset();

      _file_itm_data._current_input_pass++;
      auto const num_passes = _file_itm_data.input_pass_row_group_offsets.size() - 1;
      // no more passes. we are absolutely done with this file.
      if(_file_itm_data._current_input_pass == num_passes){
        return;
      }

      // preprocess the next pass
      preprocess_next_pass();
    }
  } 
  
  // next pass
  pass.subpass = std::make_unique<subpass_intermediate_data>();
  auto& subpass = *pass.subpass;

  auto const num_columns = pass.page_offsets.size() - 1;
  
  auto [page_indices, total_pages] = [&]() -> std::pair<std::vector<page_span>, size_t> {    
    // special case:  if we contain no compressed data, or if we have no input limit, we can always just do 1 subpass since
    // what we already have loaded is all the temporary memory we will ever use.
    if(!pass.has_compressed_data || _input_pass_read_limit == 0){
      std::vector<page_span> page_indices;
      page_indices.reserve(num_columns);
      auto iter = thrust::make_counting_iterator(0);
      std::transform(iter, iter + num_columns, std::back_inserter(page_indices), [&](size_t i) -> page_span {
        return {static_cast<size_t>(pass.page_offsets[i]), static_cast<size_t>(pass.page_offsets[i+1])};
      });
      return {page_indices, pass.pages.size()};
    } 
    // otherwise we have to look forward and choose a batch of pages

    // generate cumulative page sizes.
    rmm::device_uvector<cumulative_page_info> c_info(pass.pages.size(), _stream);
    auto page_keys = make_page_key_iterator(pass.pages);
    auto page_size = thrust::make_transform_iterator(pass.pages.d_begin(), get_page_size{pass.chunks});
    thrust::inclusive_scan_by_key(rmm::exec_policy(_stream),
                                  page_keys,
                                  page_keys + pass.pages.size(),
                                  page_size,
                                  c_info.begin(),
                                  thrust::equal_to{},
                                  cumulative_page_sum{});
    // print_cumulative_page_info(pass.pages, c_info, _stream);

    // get the next batch of pages
    return compute_next_subpass(c_info, pass.pages, pass.page_offsets, pass.processed_rows, _input_pass_read_limit, num_columns, _stream);
  }();
  
  // fill out the subpass struct  
  subpass.pages = cudf::detail::hostdevice_vector<PageInfo>(0, total_pages, _stream);
  subpass.page_src_index = cudf::detail::hostdevice_vector<size_t>(total_pages, total_pages, _stream);
  // copy the appropriate subset of pages from each column
  size_t page_count = 0;
  for(size_t c_idx=0; c_idx<num_columns; c_idx++){
    auto const num_column_pages = page_indices[c_idx].end - page_indices[c_idx].start;
    subpass.chunk_page_count.push_back(num_column_pages);
    std::copy(pass.pages.begin() + page_indices[c_idx].start,
              pass.pages.begin() + page_indices[c_idx].end,
              std::back_inserter(subpass.pages));
    
    // mapping back to original pages in the pass
    thrust::sequence(thrust::host,
                     subpass.page_src_index.begin() + page_count,
                     subpass.page_src_index.begin() + page_count + num_column_pages,
                     page_indices[c_idx].start);
    page_count += num_column_pages;
  }
  subpass.pages.host_to_device_async(_stream);
  subpass.page_src_index.host_to_device_async(_stream);

  //print_hostdevice_vector(subpass.page_src_index);

  // decompress the pages
  if (pass.has_compressed_data) {
    subpass.decomp_page_data = decompress_page_data(pass.chunks, subpass.pages, _stream);
    /*
    // Free compressed data
    for (size_t c = 0; c < chunks.size(); c++) {
      if (chunks[c].codec != Compression::UNCOMPRESSED) { raw_page_data[c].reset(); }
    }
    */
  }
  // buffers needed by the decode kernels
  subpass.pages.device_to_host_sync(_stream);
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
  preprocess_subpass_pages(uses_custom_row_bounds, _output_chunk_read_limit);
}

}  // namespace cudf::io::parquet::detail
