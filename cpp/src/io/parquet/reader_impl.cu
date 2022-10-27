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

#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace cudf::io::detail::parquet {

namespace {

struct get_page_nesting_size {
  size_type const src_col_schema;
  size_type const depth;
  gpu::PageInfo const* const pages;

  __device__ size_type operator()(int index) const
  {
    auto const& page = pages[index];
    if (page.src_col_schema != src_col_schema || page.flags & gpu::PAGEINFO_FLAGS_DICTIONARY) {
      return 0;
    }
    return page.nesting[depth].size;
  }
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

void reader::impl::allocate_columns(hostdevice_vector<gpu::ColumnChunkDesc>& chunks,
                                    hostdevice_vector<gpu::PageInfo>& pages,
                                    gpu::chunk_intermediate_data const& id,
                                    size_t min_row,
                                    size_t num_rows,
                                    bool uses_custom_row_bounds)
{
  // computes:
  // PageNestingInfo::size for each level of nesting, for each page, taking row bounds into account.
  // PageInfo::skipped_values, which tells us where to start decoding in the input to respect the
  // user bounds.
  // It is only necessary to do this second pass if uses_custom_row_bounds is set (if the user has
  // specified artifical bounds).
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

      auto* cols = &_output_buffers;
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
    auto* cols = &_output_buffers;
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

    auto* cols = &_output_buffers;
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

    auto* cols = &_output_buffers;
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
  std::tie(_input_columns, _output_buffers, _output_column_schemas) =
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
  for (auto const& buff : _output_buffers) {
    auto& new_buff =
      _output_buffers_template.emplace_back(column_buffer(buff.type, buff.is_nullable));
    copy_output_buffer(buff, new_buff);
  }
}

void reader::impl::prepare_data(size_type skip_rows,
                                size_type num_rows,
                                bool uses_custom_row_bounds,
                                std::vector<std::vector<size_type>> const& row_group_list)
{
  if (_file_preprocessed) { return; }

  const auto [skip_rows_corrected, num_rows_corrected, row_groups_info] =
    _metadata->select_row_groups(row_group_list, skip_rows, num_rows);

  if (num_rows_corrected > 0 && row_groups_info.size() != 0 && _input_columns.size() != 0) {
    load_and_decompress_data(row_groups_info, num_rows_corrected);

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
  out_columns.reserve(_output_buffers.size());

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
  for (size_t i = 0; i < _output_buffers.size(); ++i) {
    auto const metadata = _reader_column_schema.has_value()
                            ? std::make_optional<reader_column_schema>((*_reader_column_schema)[i])
                            : std::nullopt;
    // Only construct `out_metadata` if `_output_metadata` has not been cached.
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info.emplace_back("");
      out_columns.emplace_back(make_column(_output_buffers[i], &col_name, metadata, _stream, _mr));
    } else {
      out_columns.emplace_back(make_column(_output_buffers[i], nullptr, metadata, _stream, _mr));
    }
  }

  return finalize_output(out_metadata, out_columns);
}

table_with_metadata reader::impl::finalize_output(table_metadata& out_metadata,
                                                  std::vector<std::unique_ptr<column>>& out_columns)
{
  // Create empty columns as needed (this can happen if we've ended up with no actual data to read)
  for (size_t i = out_columns.size(); i < _output_buffers.size(); ++i) {
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info.emplace_back("");
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], &col_name, _stream, _mr));
    } else {
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], nullptr, _stream, _mr));
    }
  }

  if (!_output_metadata) {
    // Return column names (must match order of returned columns)
    out_metadata.column_names.resize(_output_buffers.size());
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
  prepare_data(
    skip_rows, num_rows, uses_custom_row_bounds || _chunk_read_limit > 0, row_group_list);
  return read_chunk_internal(uses_custom_row_bounds || _chunk_read_limit > 0);
#else
  CUDF_EXPECTS(_chunk_read_limit == 0, "Reading the whole file must not have non-zero byte_limit.");
  prepare_data(skip_rows, num_rows, uses_custom_row_bounds, row_group_list);
  return read_chunk_internal(uses_custom_row_bounds);
#endif
}

table_with_metadata reader::impl::read_chunk()
{
  // Reset the output buffers to their original states (right after reader construction).
  _output_buffers.resize(0);
  for (auto const& buff : _output_buffers_template) {
    auto& new_buff = _output_buffers.emplace_back(column_buffer(buff.type, buff.is_nullable));
    copy_output_buffer(buff, new_buff);
  }

  prepare_data(0, -1, true, {});
  return read_chunk_internal(true);
}

bool reader::impl::has_next()
{
  prepare_data(0, -1, true, {});
  return _current_read_chunk < _chunk_read_info.size();
}

}  // namespace cudf::io::detail::parquet
