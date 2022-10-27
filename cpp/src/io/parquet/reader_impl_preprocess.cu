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
#include "reader_impl_helpers.cuh"

#include <io/utilities/time_utils.cuh>

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace cudf::io::detail::parquet {

namespace {

#if defined(PREPROCESS_DEBUG)
void print_pages(hostdevice_vector<gpu::PageInfo>& pages, rmm::cuda_stream_view _stream)
{
  pages.device_to_host(_stream, true);
  for (size_t idx = 0; idx < pages.size(); idx++) {
    auto const& p = pages[idx];
    // skip dictionary pages
    if (p.flags & gpu::PAGEINFO_FLAGS_DICTIONARY) { continue; }
    printf(
      "P(%lu, s:%d): chunk_row(%d), num_rows(%d), skipped_values(%d), skipped_leaf_values(%d)\n",
      idx,
      p.src_col_schema,
      p.chunk_row,
      p.num_rows,
      p.skipped_values,
      p.skipped_leaf_values);
  }
}

void print_chunks(hostdevice_vector<gpu::ColumnChunkDesc>& chunks, rmm::cuda_stream_view _stream)
{
  chunks.device_to_host(_stream, true);
  for (size_t idx = 0; idx < chunks.size(); idx++) {
    auto const& c = chunks[idx];
    printf("C(%lu, s:%d): num_values(%lu), start_row(%lu), num_rows(%u)\n",
           idx,
           c.src_col_schema,
           c.num_values,
           c.start_row,
           c.num_rows);
  }
}
#endif  // PREPROCESS_DEBUG

struct cumulative_row_info {
  size_t row_count;   // cumulative row count
  size_t size_bytes;  // cumulative size in bytes
  int key;            // schema index
};
struct cumulative_row_sum {
  cumulative_row_info operator()
    __device__(cumulative_row_info const& a, cumulative_row_info const& b) const
  {
    return cumulative_row_info{a.row_count + b.row_count, a.size_bytes + b.size_bytes, a.key};
  }
};

struct row_size_functor {
  __device__ size_t validity_size(size_t num_rows, bool nullable)
  {
    return nullable ? (cudf::util::div_rounding_up_safe(num_rows, size_t{32}) / 8) : 0;
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
  auto const offset_size = sizeof(offset_type);
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
  auto const offset_size = sizeof(offset_type);
  return (offset_size * (num_rows + 1)) + validity_size(num_rows, nullable);
}

struct get_cumulative_row_info {
  gpu::PageInfo const* const pages;

  cumulative_row_info operator() __device__(size_type index)
  {
    auto const& page = pages[index];
    if (page.flags & gpu::PAGEINFO_FLAGS_DICTIONARY) {
      return cumulative_row_info{0, 0, page.src_col_schema};
    }
    size_t const row_count = page.nesting[0].size;
    return cumulative_row_info{
      row_count,
      // note: the size of the actual char bytes for strings is tracked in the `str_bytes` field, so
      // the row_size_functor{} itself is only returning the size of offsets+validity
      cudf::type_dispatcher(data_type{page.type}, row_size_functor{}, row_count, false) +
        page.str_bytes,
      page.src_col_schema};
  }
};

struct row_total_size {
  cumulative_row_info const* const c_info;
  size_type const* const key_offsets;
  size_t const num_keys;

  __device__ cumulative_row_info operator()(cumulative_row_info const& i)
  {
    // sum sizes for each input column at this row
    size_t sum = 0;
    for (int idx = 0; idx < num_keys; idx++) {
      auto const start = key_offsets[idx];
      auto const end   = key_offsets[idx + 1];
      auto iter        = cudf::detail::make_counting_transform_iterator(
        0, [&] __device__(size_type i) { return c_info[start + i].row_count; });
      auto const page_index =
        (thrust::lower_bound(thrust::seq, iter, iter + (end - start), i.row_count) - iter) + start;
      // printf("KI(%d): start(%d), end(%d), page_index(%d), size_bytes(%lu)\n", idx, start, end,
      // (int)page_index, c_info[page_index].size_bytes);
      sum += c_info[page_index].size_bytes;
    }
    return {i.row_count, sum};
  }
};

std::vector<gpu::chunk_read_info> find_splits(std::vector<cumulative_row_info> const& sizes,
                                              size_type num_rows,
                                              size_t chunked_read_size)
{
  // now we have an array of {row_count, real output bytes}. just walk through it and generate
  // splits.
  // TODO: come up with a clever way to do this entirely in parallel. For now, as long as batch
  // sizes are reasonably large, this shouldn't iterate too many times
  std::vector<gpu::chunk_read_info> splits;
  {
    size_t cur_pos             = 0;
    size_t cur_cumulative_size = 0;
    size_t cur_row_count       = 0;
    auto start = thrust::make_transform_iterator(sizes.begin(), [&](cumulative_row_info const& i) {
      return i.size_bytes - cur_cumulative_size;
    });
    auto end   = start + sizes.size();
    while (cur_row_count < static_cast<size_t>(num_rows)) {
      int64_t p = thrust::lower_bound(
                    thrust::seq, start + cur_pos, end, static_cast<size_t>(chunked_read_size)) -
                  start;

      // if we're past the end, or if the returned bucket is > than the chunked_read_size, move back
      // one.
      if (static_cast<size_t>(p) >= sizes.size() ||
          (sizes[p].size_bytes - cur_cumulative_size > static_cast<size_t>(chunked_read_size))) {
        p--;
      }

      // best-try. if we can't find something that'll fit, we have to go bigger. we're doing this in
      // a loop because all of the cumulative sizes for all the pages are sorted into one big list.
      // so if we had two columns, both of which had an entry {1000, 10000}, that entry would be in
      // the list twice. so we have to iterate until we skip past all of them.  The idea is that we
      // either do this, or we have to call unique() on the input first.
      while (p < (static_cast<int64_t>(sizes.size()) - 1) &&
             (sizes[p].row_count == cur_row_count || p < 0)) {
        p++;
      }

      auto const start_row = cur_row_count;
      cur_row_count        = sizes[p].row_count;
      splits.push_back(gpu::chunk_read_info{start_row, cur_row_count - start_row});
      // printf("Split: {%lu, %lu}\n", splits.back().skip_rows, splits.back().num_rows);
      cur_pos             = p;
      cur_cumulative_size = sizes[p].size_bytes;
    }
  }
  return splits;
}

std::vector<gpu::chunk_read_info> compute_splits(hostdevice_vector<gpu::PageInfo>& pages,
                                                 gpu::chunk_intermediate_data const& id,
                                                 size_type num_rows,
                                                 size_type chunked_read_size,
                                                 rmm::cuda_stream_view stream)
{
  auto const& page_keys  = id.page_keys;
  auto const& page_index = id.page_index;

  // generate cumulative row counts and sizes
  rmm::device_uvector<cumulative_row_info> c_info(page_keys.size(), stream);
  // convert PageInfo to cumulative_row_info
  auto page_input = thrust::make_transform_iterator(page_index.begin(),
                                                    get_cumulative_row_info{pages.device_ptr()});
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                page_keys.begin(),
                                page_keys.end(),
                                page_input,
                                c_info.begin(),
                                thrust::equal_to{},
                                cumulative_row_sum{});
  // clang-format off
  /*
  stream.synchronize();
  pages.device_to_host(stream);
  std::vector<int> h_page_index(pages.size());
  cudaMemcpyAsync(h_page_index.data(), page_index.begin(), sizeof(int) * pages.size(), cudaMemcpyDeviceToHost, stream);
  stream.synchronize();
  for(size_t idx=0; idx<pages.size(); idx++){
    auto const& page = pages[h_page_index[idx]];
    if(page.flags & gpu::PAGEINFO_FLAGS_DICTIONARY){
      continue;
    }
    gpu::PageNestingInfo pni;
    cudaMemcpy(&pni, &page.nesting[0], sizeof(gpu::PageNestingInfo), cudaMemcpyDeviceToHost);
    printf("P(%lu): schema(%d), num_rows(%d), nesting size 0(%d), str_bytes(%d)\n", idx, page.src_col_schema, page.num_rows, pni.size, page.str_bytes);
  }
  printf("---------\n");
  std::vector<cumulative_row_info> h_c_info(page_keys.size());
  cudaMemcpy(h_c_info.data(), c_info.data(), sizeof(cumulative_row_info) * page_keys.size(), cudaMemcpyDeviceToHost);
  for(size_t idx=0; idx<page_keys.size(); idx++){
    printf("C(%lu): %lu, %lu\n", idx, h_c_info[idx].row_count, h_c_info[idx].size_bytes);
  }
  */
  // clang-format on

  // sort by row count
  rmm::device_uvector<cumulative_row_info> c_info_sorted{c_info, stream};
  thrust::sort(rmm::exec_policy(stream),
               c_info_sorted.begin(),
               c_info_sorted.end(),
               [] __device__(cumulative_row_info const& a, cumulative_row_info const& b) {
                 return a.row_count < b.row_count;
               });

  // generate key offsets (offsets to the start of each partition of keys). worst case is 1 page per
  // key
  rmm::device_uvector<size_type> key_offsets(page_keys.size() + 1, stream);
  auto [_, key_offsets_end]    = thrust::reduce_by_key(rmm::exec_policy(stream),
                                                    page_keys.begin(),
                                                    page_keys.end(),
                                                    thrust::make_constant_iterator(1),
                                                    thrust::make_discard_iterator(),
                                                    key_offsets.begin());
  size_t const num_unique_keys = key_offsets_end - key_offsets.begin();
  // clang-format off
  /*
  stream.synchronize();
  printf("Num keys: %d\n", (int)num_unique_keys);
  std::vector<size_type> h_key_offsets(num_unique_keys);
  cudaMemcpy(h_key_offsets.data(), key_offsets.data(), sizeof(size_type) * num_unique_keys, cudaMemcpyDeviceToHost);
  for(size_t idx=0; idx<num_unique_keys; idx++){
    printf("Offset sizes(%lu): %d\n", idx, h_key_offsets[idx]);
  }
  */
  // clang-format on

  thrust::exclusive_scan(
    rmm::exec_policy(stream), key_offsets.begin(), key_offsets.end(), key_offsets.begin());
  // clang-format off
  /*
  stream.synchronize();
  cudaMemcpy(h_key_offsets.data(), key_offsets.data(), sizeof(size_type) * (num_unique_keys + 1), cudaMemcpyDeviceToHost);
  for(size_t idx=0; idx<num_unique_keys+1; idx++){
    printf("Offset values(%lu): %d\n", idx, h_key_offsets[idx]);
  }
  */
  // clang-format on

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
  rmm::device_uvector<cumulative_row_info> adjusted(c_info.size(), stream);
  thrust::transform(rmm::exec_policy(stream),
                    c_info_sorted.begin(),
                    c_info_sorted.end(),
                    adjusted.begin(),
                    row_total_size{c_info.data(), key_offsets.data(), num_unique_keys});

  // bring back to the cpu
  std::vector<cumulative_row_info> h_adjusted(adjusted.size());
  cudaMemcpyAsync(h_adjusted.data(),
                  adjusted.data(),
                  sizeof(cumulative_row_info) * c_info.size(),
                  cudaMemcpyDeviceToHost,
                  stream);
  stream.synchronize();
  // clang-format off
  /*
  for(size_t idx=0; idx<h_adjusted.size(); idx++){
    printf("A(%lu): %lu, %lu\n", idx, h_adjusted[idx].row_count, h_adjusted[idx].size_bytes);
  }
  */
  // clang-format on

  return find_splits(h_adjusted, num_rows, chunked_read_size);
}

struct get_page_chunk_idx {
  __device__ size_type operator()(gpu::PageInfo const& page) { return page.chunk_idx; }
};

struct get_page_num_rows {
  __device__ size_type operator()(gpu::PageInfo const& page) { return page.num_rows; }
};

struct get_page_schema {
  __device__ size_type operator()(gpu::PageInfo const& page) { return page.src_col_schema; }
};

struct get_page_nesting_size {
  size_type const src_col_schema;
  size_type const depth;
  gpu::PageInfo const* const pages;

  __device__ size_type operator()(int index)
  {
    auto const& page = pages[index];
    if (page.src_col_schema != src_col_schema || page.flags & gpu::PAGEINFO_FLAGS_DICTIONARY) {
      return 0;
    }
    return page.nesting[depth].batch_size;
  }
};

struct chunk_row_output_iter {
  gpu::PageInfo* p;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  __host__ __device__ chunk_row_output_iter operator+(int i)
  {
    return chunk_row_output_iter{p + i};
  }

  __host__ __device__ void operator++() { p++; }

  __device__ reference operator[](int i) { return p[i].chunk_row; }
  __device__ reference operator*() { return p->chunk_row; }
  // __device__ void operator=(value_type v) { p->chunk_row = v; }
};

struct start_offset_output_iterator {
  gpu::PageInfo* pages;
  int const* page_indices;
  int cur_index;
  int src_col_schema;
  int nesting_depth;
  int empty               = 0;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type*;
  using reference         = size_type&;
  using iterator_category = thrust::output_device_iterator_tag;

  __host__ __device__ void operator=(start_offset_output_iterator const& other)
  {
    pages          = other.pages;
    page_indices   = other.page_indices;
    cur_index      = other.cur_index;
    src_col_schema = other.src_col_schema;
    nesting_depth  = other.nesting_depth;
  }

  __host__ __device__ start_offset_output_iterator operator+(int i)
  {
    return start_offset_output_iterator{
      pages, page_indices, cur_index + i, src_col_schema, nesting_depth};
  }

  __host__ __device__ void operator++() { cur_index++; }

  __device__ reference operator[](int i) { return dereference(cur_index + i); }
  __device__ reference operator*() { return dereference(cur_index); }

 private:
  __device__ reference dereference(int index)
  {
    gpu::PageInfo const& p = pages[page_indices[index]];
    if (p.src_col_schema != src_col_schema || p.flags & gpu::PAGEINFO_FLAGS_DICTIONARY) {
      return empty;
    }
    return p.nesting[nesting_depth].page_start_value;
  }
};

}  // anonymous namespace

void reader::impl::preprocess_columns(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                      hostdevice_vector<gpu::PageInfo>& pages,
                                      size_t min_row,
                                      size_t num_rows,
                                      bool uses_custom_row_bounds,
                                      size_type chunked_read_size)
{
  // iterate over all input columns and determine if they contain lists so we can further
  // preprocess them.
  bool has_lists = false;
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    auto const& input_col  = _input_columns[idx];
    size_t const max_depth = input_col.nesting_depth();

    auto* cols = &_output_columns;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // if this has a list parent, we will have to do further work in gpu::PreprocessColumnData
      // to know how big this buffer actually is.
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) {
        has_lists = true;
        break;
      }
    }
    if (has_lists) { break; }
  }

  // generate string dict indices if necessary
  {
    auto is_dict_chunk = [](const gpu::ColumnChunkDesc& chunk) {
      return (chunk.data_type & 0x7) == BYTE_ARRAY && chunk.num_dict_pages > 0;
    };

    // Count the number of string dictionary entries
    // NOTE: Assumes first page in the chunk is always the dictionary page
    size_t total_str_dict_indexes = 0;
    for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
      if (is_dict_chunk(chunks[c])) {
        total_str_dict_indexes += pages[page_count].num_input_values;
      }
      page_count += chunks[c].max_num_pages;
    }

    // Build index for string dictionaries since they can't be indexed
    // directly due to variable-sized elements
    _chunk_itm_data.str_dict_index =
      cudf::detail::make_zeroed_device_uvector_async<string_index_pair>(total_str_dict_indexes,
                                                                        _stream);

    // Update chunks with pointers to string dict indices
    for (size_t c = 0, page_count = 0, str_ofs = 0; c < chunks.size(); c++) {
      input_column_info const& input_col = _input_columns[chunks[c].src_col_index];
      CUDF_EXPECTS(input_col.schema_idx == chunks[c].src_col_schema,
                   "Column/page schema index mismatch");
      if (is_dict_chunk(chunks[c])) {
        chunks[c].str_dict_index = _chunk_itm_data.str_dict_index.data() + str_ofs;
        str_ofs += pages[page_count].num_input_values;
      }

      // column_data_base will always point to leaf data, even for nested types.
      page_count += chunks[c].max_num_pages;
    }

    if (total_str_dict_indexes > 0) {
      chunks.host_to_device(_stream);
      gpu::BuildStringDictionaryIndex(chunks.device_ptr(), chunks.size(), _stream);
    }
  }

  // intermediate data we will need for further chunked reads
  if (has_lists || chunked_read_size > 0) {
    // computes:
    // PageNestingInfo::num_rows for each page. the true number of rows (taking repetition into
    // account), not just the number of values. PageNestingInfo::size for each level of nesting, for
    // each page.
    //
    // we will be applying a later "trim" pass if skip_rows/num_rows is being used, which can happen
    // if:
    // - user has passed custom row bounds
    // - if we will be doing a chunked read
    gpu::ComputePageSizes(pages,
                          chunks,
                          0,  // 0-max size_t. process all possible rows
                          std::numeric_limits<size_t>::max(),
                          true,                   // compute num_rows
                          chunked_read_size > 0,  // compute string sizes
                          _stream);

    // computes:
    // PageInfo::chunk_row (the absolute start row index) for all pages
    // Note: this is doing some redundant work for pages in flat hierarchies.  chunk_row has already
    // been computed during header decoding. the overall amount of work here is very small though.
    auto key_input  = thrust::make_transform_iterator(pages.device_ptr(), get_page_chunk_idx{});
    auto page_input = thrust::make_transform_iterator(pages.device_ptr(), get_page_num_rows{});
    thrust::exclusive_scan_by_key(rmm::exec_policy(_stream),
                                  key_input,
                                  key_input + pages.size(),
                                  page_input,
                                  chunk_row_output_iter{pages.device_ptr()});

    // compute page ordering.
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
    // To use exclusive_scan_by_key, the ordering we actually want is
    //
    // 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
    //
    // We also need to preserve key-relative page ordering, so we need to use a stable sort.
    _chunk_itm_data.page_keys  = rmm::device_uvector<int>(pages.size(), _stream);
    _chunk_itm_data.page_index = rmm::device_uvector<int>(pages.size(), _stream);
    auto& page_keys            = _chunk_itm_data.page_keys;
    auto& page_index           = _chunk_itm_data.page_index;
    {
      thrust::transform(rmm::exec_policy(_stream),
                        pages.device_ptr(),
                        pages.device_ptr() + pages.size(),
                        page_keys.begin(),
                        get_page_schema{});

      thrust::sequence(rmm::exec_policy(_stream), page_index.begin(), page_index.end());
      thrust::stable_sort_by_key(rmm::exec_policy(_stream),
                                 page_keys.begin(),
                                 page_keys.end(),
                                 page_index.begin(),
                                 thrust::less<int>());
    }

    // retrieve pages back
    pages.device_to_host(_stream, true);

    // print_pages(pages, _stream);
  }

  // compute splits if necessary.
  _chunk_read_info =
    chunked_read_size > 0
      ? compute_splits(pages, _chunk_itm_data, num_rows, chunked_read_size, _stream)
      : std::vector<gpu::chunk_read_info>{{min_row, num_rows}};
}

void reader::impl::allocate_columns(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                    hostdevice_vector<gpu::PageInfo>& pages,
                                    gpu::chunk_intermediate_data const& id,
                                    size_t min_row,
                                    size_t num_rows,
                                    bool uses_custom_row_bounds)
{
  // computes:
  // PageNestingInfo::batch_size for each level of nesting, for each page, taking row bounds into
  // account. PageInfo::skipped_values, which tells us where to start decoding in the input to
  // respect the user bounds. It is only necessary to do this second pass if uses_custom_row_bounds
  // is set (if the user has specified artifical bounds).
  if (uses_custom_row_bounds) {
    gpu::ComputePageSizes(pages,
                          chunks,
                          min_row,
                          num_rows,
                          false,  // num_rows is already computed
                          false,  // no need to compute string sizes
                          _stream);
    // print_pages(pages, _stream);
  }

  // iterate over all input columns and allocate any associated output
  // buffers if they are not part of a list hierarchy. mark down
  // if we have any list columns that need further processing.
  bool has_lists = false;
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    auto const& input_col  = _input_columns[idx];
    size_t const max_depth = input_col.nesting_depth();

    auto* cols = &_output_columns;
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
          out_buf.type.id() == type_id::LIST && l_idx < max_depth ? num_rows + 1 : num_rows,
          _stream,
          _mr);
      }
    }
  }

  // compute output column sizes by examining the pages of the -input- columns
  if (has_lists) {
    auto& page_keys  = _chunk_itm_data.page_keys;
    auto& page_index = _chunk_itm_data.page_index;
    for (size_t idx = 0; idx < _input_columns.size(); idx++) {
      auto const& input_col = _input_columns[idx];
      auto src_col_schema   = input_col.schema_idx;
      size_t max_depth      = input_col.nesting_depth();

      auto* cols = &_output_columns;
      for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
        auto& out_buf = (*cols)[input_col.nesting[l_idx]];
        cols          = &out_buf.children;

        // size iterator. indexes pages by sorted order
        auto size_input = thrust::make_transform_iterator(
          page_index.begin(),
          get_page_nesting_size{src_col_schema, static_cast<size_type>(l_idx), pages.device_ptr()});

        // if this buffer is part of a list hierarchy, we need to determine it's
        // final size and allocate it here.
        //
        // for struct columns, higher levels of the output columns are shared between input
        // columns. so don't compute any given level more than once.
        if ((out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) && out_buf.size == 0) {
          int size =
            thrust::reduce(rmm::exec_policy(_stream), size_input, size_input + pages.size());

          // if this is a list column add 1 for non-leaf levels for the terminating offset
          if (out_buf.type.id() == type_id::LIST && l_idx < max_depth) { size++; }

          // allocate
          out_buf.create(size, _stream, _mr);
        }

        // for nested hierarchies, compute per-page start offset
        if (input_col.has_repetition) {
          thrust::exclusive_scan_by_key(
            rmm::exec_policy(_stream),
            page_keys.begin(),
            page_keys.end(),
            size_input,
            start_offset_output_iterator{pages.device_ptr(),
                                         page_index.begin(),
                                         0,
                                         static_cast<int>(src_col_schema),
                                         static_cast<int>(l_idx)});
        }
      }
    }
  }
}

namespace {
/**
 * @brief Function that returns the required the number of bits to store a value
 */
template <typename T = uint8_t>
T required_bits(uint32_t max_level)
{
  return static_cast<T>(CompactProtocolReader::NumRequiredBits(max_level));
}

/**
 * @brief Converts cuDF units to Parquet units.
 *
 * @return A tuple of Parquet type width, Parquet clock rate and Parquet decimal type.
 */
std::tuple<int32_t, int32_t, int8_t> conversion_info(type_id column_type_id,
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

}  // namespace

std::pair<size_type, size_type> reader::impl::preprocess_file(
  size_type skip_rows,
  size_type num_rows,
  const std::vector<std::vector<size_type>>& row_group_list)
{
  //  printf("\n\n\n\npreprocess========================\n");

  // Select only row groups required
  // Note: `skip_rows` and `num_rows` will be modified in this function.
  const auto selected_row_groups =
    _metadata->select_row_groups(row_group_list, skip_rows, num_rows);

  // TODO: fix this
  if (selected_row_groups.size() == 0 || _input_columns.size() == 0) {
    return {skip_rows, num_rows};
  }

  // TODO: fix this.
  // Need to check if the file actually has data.
  _file_itm_data.has_data = true;

  // Descriptors for all the chunks that make up the selected columns
  const auto num_input_columns = _input_columns.size();
  const auto num_chunks        = selected_row_groups.size() * num_input_columns;
  _file_itm_data.chunks        = hostdevice_vector<gpu::ColumnChunkDesc>(0, num_chunks, _stream);

  // Association between each column chunk and its source
  std::vector<size_type> chunk_source_map(num_chunks);

  // Tracker for eventually deallocating compressed and uncompressed data
  _file_itm_data.raw_page_data = std::vector<std::unique_ptr<datasource::buffer>>(num_chunks);

  // Keep track of column chunk file offsets
  std::vector<size_t> column_chunk_offsets(num_chunks);

  // Initialize column chunk information
  size_t total_decompressed_size = 0;
  auto remaining_rows            = num_rows;
  std::vector<std::future<void>> read_rowgroup_tasks;
  for (const auto& rg : selected_row_groups) {
    const auto& row_group       = _metadata->get_row_group(rg.index, rg.source_index);
    auto const row_group_start  = rg.start_row;
    auto const row_group_source = rg.source_index;
    auto const row_group_rows   = std::min<int>(remaining_rows, row_group.num_rows);
    auto const io_chunk_idx     = _file_itm_data.chunks.size();

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

      column_chunk_offsets[_file_itm_data.chunks.size()] =
        (col_meta.dictionary_page_offset != 0)
          ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
          : col_meta.data_page_offset;

      _file_itm_data.chunks.push_back(
        gpu::ColumnChunkDesc(col_meta.total_compressed_size,
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
      chunk_source_map[_file_itm_data.chunks.size() - 1] = row_group_source;

      if (col_meta.codec != Compression::UNCOMPRESSED) {
        total_decompressed_size += col_meta.total_uncompressed_size;
      }
    }
    // Read compressed chunk data to device memory
    read_rowgroup_tasks.push_back(read_column_chunks(_file_itm_data.raw_page_data,
                                                     _file_itm_data.chunks,
                                                     io_chunk_idx,
                                                     _file_itm_data.chunks.size(),
                                                     column_chunk_offsets,
                                                     chunk_source_map));

    remaining_rows -= row_group.num_rows;
  }
  for (auto& task : read_rowgroup_tasks) {
    task.wait();
  }
  assert(remaining_rows <= 0);

  // Process dataset chunk pages into output columns
  const auto total_pages    = count_page_headers(_file_itm_data.chunks);
  _file_itm_data.pages_info = hostdevice_vector<gpu::PageInfo>(total_pages, total_pages, _stream);

  if (total_pages > 0) {
    // decoding of column/page information
    decode_page_headers(_file_itm_data.chunks, _file_itm_data.pages_info);
    if (total_decompressed_size > 0) {
      _file_itm_data.decomp_page_data =
        decompress_page_data(_file_itm_data.chunks, _file_itm_data.pages_info);
      // Free compressed data
      for (size_t c = 0; c < _file_itm_data.chunks.size(); c++) {
        if (_file_itm_data.chunks[c].codec != parquet::Compression::UNCOMPRESSED) {
          _file_itm_data.raw_page_data[c].reset();
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

    allocate_nesting_info(
      _file_itm_data.chunks, _file_itm_data.pages_info, _file_itm_data.page_nesting_info);
  }

  return {skip_rows, num_rows};
}

}  // namespace cudf::io::detail::parquet
