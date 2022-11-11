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

#include <numeric>

namespace cudf::io::detail::parquet {

void reader::impl::decode_page_data(size_t skip_rows, size_t num_rows)
{
  auto& chunks       = _file_itm_data.chunks;
  auto& pages        = _file_itm_data.pages_info;
  auto& page_nesting = _file_itm_data.page_nesting_info;

  auto is_dict_chunk = [](const gpu::ColumnChunkDesc& chunk) {
    return (chunk.data_type & 0x7) == BYTE_ARRAY && chunk.num_dict_pages > 0;
  };

  // Count the number of string dictionary entries
  // NOTE: Assumes first page in the chunk is always the dictionary page
  size_t total_str_dict_indexes = 0;
  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    if (is_dict_chunk(chunks[c])) { total_str_dict_indexes += pages[page_count].num_input_values; }
    page_count += chunks[c].max_num_pages;
  }

  // Build index for string dictionaries since they can't be indexed
  // directly due to variable-sized elements
  auto str_dict_index = cudf::detail::make_zeroed_device_uvector_async<string_index_pair>(
    total_str_dict_indexes, _stream);

  size_t const sum_max_depths = std::accumulate(
    chunks.begin(), chunks.end(), 0, [&](size_t cursum, gpu::ColumnChunkDesc const& chunk) {
      return cursum + _metadata->get_output_nesting_depth(chunk.src_col_schema);
    });

  // In order to reduce the number of allocations of hostdevice_vector, we allocate a single vector
  // to store all per-chunk pointers to nested data/nullmask. `chunk_offsets[i]` will store the
  // offset into `chunk_nested_data`/`chunk_nested_valids` for the array of pointers for chunk `i`
  auto chunk_nested_valids = hostdevice_vector<uint32_t*>(sum_max_depths, _stream);
  auto chunk_nested_data   = hostdevice_vector<void*>(sum_max_depths, _stream);
  auto chunk_offsets       = std::vector<size_t>();

  // Update chunks with pointers to column data.
  for (size_t c = 0, page_count = 0, str_ofs = 0, chunk_off = 0; c < chunks.size(); c++) {
    input_column_info const& input_col = _input_columns[chunks[c].src_col_index];
    CUDF_EXPECTS(input_col.schema_idx == chunks[c].src_col_schema,
                 "Column/page schema index mismatch");

    if (is_dict_chunk(chunks[c])) {
      chunks[c].str_dict_index = str_dict_index.data() + str_ofs;
      str_ofs += pages[page_count].num_input_values;
    }

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

  if (total_str_dict_indexes > 0) {
    gpu::BuildStringDictionaryIndex(chunks.device_ptr(), chunks.size(), _stream);
  }

  gpu::DecodePageData(pages, chunks, num_rows, skip_rows, _stream);
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

void reader::impl::prepare_data(size_type skip_rows,
                                size_type num_rows,
                                bool uses_custom_row_bounds,
                                host_span<std::vector<size_type> const> row_group_indices)
{
  const auto [skip_rows_corrected, num_rows_corrected, row_groups_info] =
    _metadata->select_row_groups(row_group_indices, skip_rows, num_rows);
  _skip_rows = skip_rows_corrected;
  _num_rows  = num_rows_corrected;

  if (num_rows_corrected > 0 && row_groups_info.size() != 0 && _input_columns.size() != 0) {
    load_and_decompress_data(row_groups_info, num_rows_corrected);
  }
}

table_with_metadata reader::impl::read_chunk_internal(bool uses_custom_row_bounds)
{
  auto out_metadata = table_metadata{};

  // output cudf columns as determined by the top level schema
  auto out_columns = std::vector<std::unique_ptr<column>>{};
  out_columns.reserve(_output_buffers.size());

  if (_num_rows == 0) { return finalize_output(out_metadata, out_columns); }

  allocate_columns(_skip_rows, _num_rows, uses_custom_row_bounds);

  decode_page_data(_skip_rows, _num_rows);

  // Create the final output cudf columns
  for (size_t i = 0; i < _output_buffers.size(); ++i) {
    auto const metadata        = _reader_column_schema.has_value()
                                   ? std::make_optional<reader_column_schema>((*_reader_column_schema)[i])
                                   : std::nullopt;
    column_name_info& col_name = out_metadata.schema_info.emplace_back("");
    out_columns.emplace_back(make_column(_output_buffers[i], &col_name, metadata, _stream, _mr));
  }

  return finalize_output(out_metadata, out_columns);
}

table_with_metadata reader::impl::finalize_output(table_metadata& out_metadata,
                                                  std::vector<std::unique_ptr<column>>& out_columns)
{
  // Create empty columns as needed (this can happen if we've ended up with no actual data to read)
  for (size_t i = out_columns.size(); i < _output_buffers.size(); ++i) {
    column_name_info& col_name = out_metadata.schema_info.emplace_back("");
    out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], &col_name, _stream, _mr));
  }

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

  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

table_with_metadata reader::impl::read(size_type skip_rows,
                                       size_type num_rows,
                                       bool uses_custom_row_bounds,
                                       host_span<std::vector<size_type> const> row_group_indices)
{
  prepare_data(skip_rows, num_rows, uses_custom_row_bounds, row_group_indices);
  return read_chunk_internal(uses_custom_row_bounds);
}

}  // namespace cudf::io::detail::parquet
