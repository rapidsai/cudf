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

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>

namespace cudf::io::detail::parquet {

// Import functionality that's independent of legacy code
using namespace cudf::io::parquet;
using namespace cudf::io;

namespace {

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
  template <typename T>
  __device__ size_t operator()(size_t num_rows, bool nullable)
  {
    auto const element_size = sizeof(device_storage_type_t<T>);
    return (element_size * num_rows) +
           (nullable ? (cudf::util::div_rounding_up_safe(num_rows, size_t{32}) / 8) : 0);
  }
};

template <>
__device__ size_t row_size_functor::operator()<list_view>(size_t num_rows, bool nullable)
{
  auto const offset_size = sizeof(offset_type);
  return (offset_size * (num_rows + 1)) +
         (nullable ? (cudf::util::div_rounding_up_safe(num_rows, size_t{32}) / 8) : 0);
}

template <>
__device__ size_t row_size_functor::operator()<struct_view>(size_t num_rows, bool nullable)
{
  return nullable ? (cudf::util::div_rounding_up_safe(num_rows, size_t{32}) / 8) : 0;
}

template <>
__device__ size_t row_size_functor::operator()<string_view>(size_t num_rows, bool nullable)
{
  // CUDF_FAIL("String types currently unsupported");
  return 0;
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
      cudf::type_dispatcher(data_type{page.nesting[0].type}, row_size_functor{}, row_count, false),
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
  /*
  stream.synchronize();
  pages.device_to_host(stream);
  std::vector<int> h_page_index(pages.size());
  cudaMemcpyAsync(h_page_index.data(), page_index.begin(), sizeof(int) * pages.size(),
  cudaMemcpyDeviceToHost, stream); stream.synchronize(); for(size_t idx=0; idx<pages.size(); idx++){
    auto const& page = pages[h_page_index[idx]];
    if(page.flags & gpu::PAGEINFO_FLAGS_DICTIONARY){
      continue;
    }
    gpu::PageNestingInfo pni;
    cudaMemcpy(&pni, &page.nesting[0], sizeof(gpu::PageNestingInfo), cudaMemcpyDeviceToHost);
    printf("P(%lu): schema(%d), num_rows(%d), nesting size 0(%d)\n", idx, page.src_col_schema,
  page.num_rows, pni.size);
  }
  printf("---------\n");
  std::vector<cumulative_row_info> h_c_info(page_keys.size());
  cudaMemcpy(h_c_info.data(), c_info.data(), sizeof(cumulative_row_info) * page_keys.size(),
  cudaMemcpyDeviceToHost); for(size_t idx=0; idx<page_keys.size(); idx++){ printf("C(%lu): %lu,
  %lu\n", idx, h_c_info[idx].row_count, h_c_info[idx].size_bytes);
  }
  */

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
  /*
  stream.synchronize();
  printf("Num keys: %d\n", (int)num_unique_keys);
  std::vector<size_type> h_key_offsets(num_unique_keys);
  cudaMemcpy(h_key_offsets.data(), key_offsets.data(), sizeof(size_type) * num_unique_keys,
  cudaMemcpyDeviceToHost); for(size_t idx=0; idx<num_unique_keys; idx++){ printf("Offset sizes(%lu):
  %d\n", idx, h_key_offsets[idx]);
  }
  */

  thrust::exclusive_scan(
    rmm::exec_policy(stream), key_offsets.begin(), key_offsets.end() + 1, key_offsets.begin());
  /*
  stream.synchronize();
  cudaMemcpy(h_key_offsets.data(), key_offsets.data(), sizeof(size_type) * (num_unique_keys + 1),
  cudaMemcpyDeviceToHost); for(size_t idx=0; idx<num_unique_keys+1; idx++){ printf("Offset
  values(%lu): %d\n", idx, h_key_offsets[idx]);
  }
  */

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
  /*
  for(size_t idx=0; idx<h_adjusted.size(); idx++){
    printf("A(%lu): %lu, %lu\n", idx, h_adjusted[idx].row_count, h_adjusted[idx].size_bytes);
  }
  */

  // now we have an array of {row_count, real output bytes}. just walk through it and generate
  // splits.
  // TODO: come up with a clever way to do this entirely in parallel. For now, as long as batch
  // sizes are reasonably large, this shouldn't iterate too many times
  std::vector<gpu::chunk_read_info> splits;
  {
    size_t cur_pos         = 0;
    size_t cumulative_size = 0;
    size_t cur_row_count   = 0;
    while (cur_row_count < static_cast<size_t>(num_rows)) {
      auto iter = thrust::make_transform_iterator(
        h_adjusted.begin() + cur_pos,
        [cumulative_size](cumulative_row_info const& i) { return i.size_bytes - cumulative_size; });
      size_type p =
        (thrust::lower_bound(
           thrust::seq, iter, iter + h_adjusted.size(), static_cast<size_t>(chunked_read_size)) -
         iter) +
        cur_pos;
      if (h_adjusted[p].size_bytes - cumulative_size > static_cast<size_t>(chunked_read_size) ||
          static_cast<size_t>(p) == h_adjusted.size()) {
        p--;
      }
      if (h_adjusted[p].row_count == cur_row_count || p < 0) {
        CUDF_FAIL("Cannot find read split boundary small enough");
      }

      auto const start_row = cur_row_count;
      cur_row_count        = h_adjusted[p].row_count;
      splits.push_back(gpu::chunk_read_info{start_row, cur_row_count - start_row});
      cur_pos         = p;
      cumulative_size = h_adjusted[p].size_bytes;
    }
  }

  return splits;
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
    return page.nesting[depth].size;
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

/**
 * @copydoc cudf::io::detail::parquet::preprocess_columns
 */
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

  // intermediate data we will need for further chunked reads
  gpu::chunk_intermediate_data id;
  if (has_lists || chunked_read_size > 0) {
    // computes:
    // PageNestingInfo::size for each level of nesting, for each page.
    // This computes the size for the entire page, not taking row bounds into account.
    /*
    gpuComputePageSizes<<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(),
      chunks,
      // if uses_custom_row_bounds is false, include all possible rows.
      uses_custom_row_bounds ? min_row : 0,
      uses_custom_row_bounds ? num_rows : INT_MAX,
      !uses_custom_row_bounds);
    */
    // we will be applying a later trim pass if skip_rows/num_rows is being used, which can happen
    // if:
    // - user has passed custom row bounds
    // - if we will be doing a chunked read
    auto const will_trim_later = uses_custom_row_bounds || chunked_read_size > 0;
    gpu::ComputePageSizes(pages,
                          chunks,
                          !will_trim_later ? min_row : 0,
                          !will_trim_later ? num_rows : INT_MAX,
                          !will_trim_later,
                          _stream);

    // computes:
    // PageInfo::chunk_row for all pages
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
    id.page_keys     = rmm::device_uvector<int>(pages.size(), _stream);
    id.page_index    = rmm::device_uvector<int>(pages.size(), _stream);
    auto& page_keys  = id.page_keys;
    auto& page_index = id.page_index;
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
  }

  // compute splits if necessary.
  std::vector<gpu::chunk_read_info> read_chunks =
    chunked_read_size > 0 ? compute_splits(pages, id, num_rows, chunked_read_size, _stream)
                          : std::vector<gpu::chunk_read_info>{{min_row, num_rows}};

  chunk_itm_data  = std::move(id);
  chunk_read_info = std::move(read_chunks);
}

void reader::impl::allocate_columns(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                    hostdevice_vector<gpu::PageInfo>& pages,
                                    gpu::chunk_intermediate_data const& id,
                                    size_t min_row,
                                    size_t num_rows,
                                    bool uses_custom_row_bounds)
{
  // computes:
  // PageNestingInfo::size for each level of nesting, for each page, taking row bounds into account.
  // PageInfo::skipped_values, which tells us where to start decoding in the input.
  // It is only necessary to do this second pass if uses_custom_row_bounds is set (if the user has
  // specified artifical bounds).
  if (uses_custom_row_bounds) {
    gpu::ComputePageSizes(pages, chunks, min_row, num_rows, true, _stream);
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
    auto& page_keys  = id.page_keys;
    auto& page_index = id.page_index;
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

        // for nested hierarchies, compute per-page start offset.
        // it would be better/safer to be checking (schema.max_repetition_level > 0) here, but
        // there's no easy way to get at that info here. we'd have to move this function into
        // reader_impl.cu
        if ((out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) ||
            out_buf.type.id() == type_id::LIST) {
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

}  // namespace cudf::io::detail::parquet
