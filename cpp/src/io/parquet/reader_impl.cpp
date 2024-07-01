/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "error.hpp"

#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>

#include <rmm/resource_ref.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <bitset>
#include <numeric>

namespace cudf::io::parquet::detail {

namespace {
// Tests the passed in logical type for a FIXED_LENGTH_BYTE_ARRAY column to see if it should
// be treated as a string. Currently the only logical type that has special handling is DECIMAL.
// Other valid types in the future would be UUID (still treated as string) and FLOAT16 (which
// for now would also be treated as a string).
inline bool is_treat_fixed_length_as_string(thrust::optional<LogicalType> const& logical_type)
{
  if (!logical_type.has_value()) { return true; }
  return logical_type->type != LogicalType::DECIMAL;
}

}  // namespace

void reader::impl::decode_page_data(read_mode mode, size_t skip_rows, size_t num_rows)
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  auto& page_nesting        = subpass.page_nesting_info;
  auto& page_nesting_decode = subpass.page_nesting_decode_info;

  auto const level_type_size = pass.level_type_size;

  // temporary space for DELTA_BYTE_ARRAY decoding. this only needs to live until
  // gpu::DecodeDeltaByteArray returns.
  rmm::device_uvector<uint8_t> delta_temp_buf(0, _stream);

  // Should not reach here if there is no page data.
  CUDF_EXPECTS(subpass.pages.size() > 0, "There are no pages to decode");

  size_t const sum_max_depths = std::accumulate(
    pass.chunks.begin(), pass.chunks.end(), 0, [&](size_t cursum, ColumnChunkDesc const& chunk) {
      return cursum + _metadata->get_output_nesting_depth(chunk.src_col_schema);
    });

  // figure out which kernels to run
  auto const kernel_mask = GetAggregatedDecodeKernelMask(subpass.pages, _stream);

  // Check to see if there are any string columns present. If so, then we need to get size info
  // for each string page. This size info will be used to pre-allocate memory for the column,
  // allowing the page decoder to write string data directly to the column buffer, rather than
  // doing a gather operation later on.
  // TODO: This step is somewhat redundant if size info has already been calculated (nested schema,
  // chunked reader).
  auto const has_strings = (kernel_mask & STRINGS_MASK) != 0;
  std::vector<size_t> col_string_sizes(_input_columns.size(), 0L);
  if (has_strings) {
    // need to compute pages bounds/sizes if we lack page indexes or are using custom bounds
    // TODO: we could probably dummy up size stats for FLBA data since we know the width
    auto const has_flba =
      std::any_of(pass.chunks.begin(), pass.chunks.end(), [](auto const& chunk) {
        return chunk.physical_type == FIXED_LEN_BYTE_ARRAY and
               is_treat_fixed_length_as_string(chunk.logical_type);
      });

    if (!_has_page_index || uses_custom_row_bounds(mode) || has_flba) {
      ComputePageStringSizes(subpass.pages,
                             pass.chunks,
                             delta_temp_buf,
                             skip_rows,
                             num_rows,
                             level_type_size,
                             kernel_mask,
                             _stream);
    }

    col_string_sizes = calculate_page_string_offsets();

    // check for overflow
    auto const threshold         = static_cast<size_t>(strings::detail::get_offset64_threshold());
    auto const has_large_strings = std::any_of(col_string_sizes.cbegin(),
                                               col_string_sizes.cend(),
                                               [=](std::size_t sz) { return sz > threshold; });
    if (has_large_strings and not strings::detail::is_large_strings_enabled()) {
      CUDF_FAIL("String column exceeds the column size limit", std::overflow_error);
    }

    // mark any chunks that are large string columns
    if (has_large_strings) {
      for (auto& chunk : pass.chunks) {
        auto const idx = chunk.src_col_index;
        if (col_string_sizes[idx] > threshold) { chunk.is_large_string_col = true; }
      }
    }
  }

  // In order to reduce the number of allocations of hostdevice_vector, we allocate a single vector
  // to store all per-chunk pointers to nested data/nullmask. `chunk_offsets[i]` will store the
  // offset into `chunk_nested_data`/`chunk_nested_valids` for the array of pointers for chunk `i`
  auto chunk_nested_valids =
    cudf::detail::hostdevice_vector<bitmask_type*>(sum_max_depths, _stream);
  auto chunk_nested_data = cudf::detail::hostdevice_vector<void*>(sum_max_depths, _stream);
  auto chunk_offsets     = std::vector<size_t>();
  auto chunk_nested_str_data =
    cudf::detail::hostdevice_vector<void*>(has_strings ? sum_max_depths : 0, _stream);

  // Update chunks with pointers to column data.
  for (size_t c = 0, chunk_off = 0; c < pass.chunks.size(); c++) {
    input_column_info const& input_col = _input_columns[pass.chunks[c].src_col_index];
    CUDF_EXPECTS(input_col.schema_idx == pass.chunks[c].src_col_schema,
                 "Column/page schema index mismatch");

    size_t max_depth = _metadata->get_output_nesting_depth(pass.chunks[c].src_col_schema);
    chunk_offsets.push_back(chunk_off);

    // get a slice of size `nesting depth` from `chunk_nested_valids` to store an array of pointers
    // to validity data
    auto valids                   = chunk_nested_valids.host_ptr(chunk_off);
    pass.chunks[c].valid_map_base = chunk_nested_valids.device_ptr(chunk_off);

    // get a slice of size `nesting depth` from `chunk_nested_data` to store an array of pointers to
    // out data
    auto data                       = chunk_nested_data.host_ptr(chunk_off);
    pass.chunks[c].column_data_base = chunk_nested_data.device_ptr(chunk_off);

    auto str_data = has_strings ? chunk_nested_str_data.host_ptr(chunk_off) : nullptr;
    pass.chunks[c].column_string_base =
      has_strings ? chunk_nested_str_data.device_ptr(chunk_off) : nullptr;

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
        // only do string buffer for leaf
        if (idx == max_depth - 1 and out_buf.string_size() == 0 and
            col_string_sizes[pass.chunks[c].src_col_index] > 0) {
          out_buf.create_string_data(col_string_sizes[pass.chunks[c].src_col_index], _stream);
        }
        if (has_strings) { str_data[idx] = out_buf.string_data(); }
        out_buf.user_data |=
          static_cast<uint32_t>(input_col.schema_idx) & PARQUET_COLUMN_BUFFER_SCHEMA_MASK;
      } else {
        valids[idx] = nullptr;
        data[idx]   = nullptr;
      }
    }
  }

  pass.chunks.host_to_device_async(_stream);
  chunk_nested_valids.host_to_device_async(_stream);
  chunk_nested_data.host_to_device_async(_stream);
  if (has_strings) { chunk_nested_str_data.host_to_device_async(_stream); }

  // create this before we fork streams
  kernel_error error_code(_stream);

  // get the number of streams we need from the pool and tell them to wait on the H2D copies
  int const nkernels = std::bitset<32>(kernel_mask).count();
  auto streams       = cudf::detail::fork_streams(_stream, nkernels);

  // launch string decoder
  int s_idx = 0;
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING) != 0) {
    DecodeStringPageData(subpass.pages,
                         pass.chunks,
                         num_rows,
                         skip_rows,
                         level_type_size,
                         error_code.data(),
                         streams[s_idx++]);
  }

  // launch delta byte array decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_BYTE_ARRAY) != 0) {
    DecodeDeltaByteArray(subpass.pages,
                         pass.chunks,
                         num_rows,
                         skip_rows,
                         level_type_size,
                         error_code.data(),
                         streams[s_idx++]);
  }

  // launch delta length byte array decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_LENGTH_BA) != 0) {
    DecodeDeltaLengthByteArray(subpass.pages,
                               pass.chunks,
                               num_rows,
                               skip_rows,
                               level_type_size,
                               error_code.data(),
                               streams[s_idx++]);
  }

  // launch delta binary decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_BINARY) != 0) {
    DecodeDeltaBinary(subpass.pages,
                      pass.chunks,
                      num_rows,
                      skip_rows,
                      level_type_size,
                      error_code.data(),
                      streams[s_idx++]);
  }

  // launch byte stream split decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::BYTE_STREAM_SPLIT_FLAT) != 0) {
    DecodeSplitPageDataFlat(subpass.pages,
                            pass.chunks,
                            num_rows,
                            skip_rows,
                            level_type_size,
                            error_code.data(),
                            streams[s_idx++]);
  }

  // launch byte stream split decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::BYTE_STREAM_SPLIT) != 0) {
    DecodeSplitPageData(subpass.pages,
                        pass.chunks,
                        num_rows,
                        skip_rows,
                        level_type_size,
                        error_code.data(),
                        streams[s_idx++]);
  }

  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_NO_DICT) != 0) {
    DecodePageDataFixed(subpass.pages,
                        pass.chunks,
                        num_rows,
                        skip_rows,
                        level_type_size,
                        error_code.data(),
                        streams[s_idx++]);
  }

  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_DICT) != 0) {
    DecodePageDataFixedDict(subpass.pages,
                            pass.chunks,
                            num_rows,
                            skip_rows,
                            level_type_size,
                            error_code.data(),
                            streams[s_idx++]);
  }

  // launch the catch-all page decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::GENERAL) != 0) {
    DecodePageData(subpass.pages,
                   pass.chunks,
                   num_rows,
                   skip_rows,
                   level_type_size,
                   error_code.data(),
                   streams[s_idx++]);
  }

  // synchronize the streams
  cudf::detail::join_streams(streams, _stream);

  subpass.pages.device_to_host_async(_stream);
  page_nesting.device_to_host_async(_stream);
  page_nesting_decode.device_to_host_async(_stream);

  if (auto const error = error_code.value_sync(_stream); error != 0) {
    CUDF_FAIL("Parquet data decode failed with code(s) " + kernel_error::to_string(error));
  }

  // for list columns, add the final offset to every offset buffer.
  // TODO : make this happen in more efficiently. Maybe use thrust::for_each
  // on each buffer.
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

      if (out_buf.type.id() == type_id::LIST &&
          (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED) == 0) {
        CUDF_EXPECTS(l_idx < input_col.nesting_depth() - 1, "Encountered a leaf list column");
        auto const& child = (*cols)[input_col.nesting[l_idx + 1]];

        // the final offset for a list at level N is the size of it's child
        size_type const offset = child.type.id() == type_id::LIST ? child.size - 1 : child.size;
        CUDF_CUDA_TRY(cudaMemcpyAsync(static_cast<size_type*>(out_buf.data()) + (out_buf.size - 1),
                                      &offset,
                                      sizeof(size_type),
                                      cudaMemcpyDefault,
                                      _stream.value()));
        out_buf.user_data |= PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED;
      } else if (out_buf.type.id() == type_id::STRING) {
        // need to cap off the string offsets column
        auto const sz = static_cast<size_type>(col_string_sizes[idx]);
        if (sz <= strings::detail::get_offset64_threshold()) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(static_cast<size_type*>(out_buf.data()) + out_buf.size,
                                        &sz,
                                        sizeof(size_type),
                                        cudaMemcpyDefault,
                                        _stream.value()));
        }
      }
    }
  }

  // update null counts in the final column buffers
  for (size_t idx = 0; idx < subpass.pages.size(); idx++) {
    PageInfo* pi = &subpass.pages[idx];
    if (pi->flags & PAGEINFO_FLAGS_DICTIONARY) { continue; }
    ColumnChunkDesc* col               = &pass.chunks[pi->chunk_idx];
    input_column_info const& input_col = _input_columns[col->src_col_index];

    int index                   = pi->nesting_decode - page_nesting_decode.device_ptr();
    PageNestingDecodeInfo* pndi = &page_nesting_decode[index];

    auto* cols = &_output_buffers;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // if I wasn't the one who wrote out the validity bits, skip it
      if (chunk_nested_valids.host_ptr(chunk_offsets[pi->chunk_idx])[l_idx] == nullptr) {
        continue;
      }
      out_buf.null_count() += pndi[l_idx].null_count;
    }
  }

  _stream.synchronize();
}

reader::impl::impl(std::vector<std::unique_ptr<datasource>>&& sources,
                   parquet_reader_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
  : impl(0 /*chunk_read_limit*/,
         0 /*input_pass_read_limit*/,
         std::forward<std::vector<std::unique_ptr<cudf::io::datasource>>>(sources),
         options,
         stream,
         mr)
{
}

reader::impl::impl(std::size_t chunk_read_limit,
                   std::size_t pass_read_limit,
                   std::vector<std::unique_ptr<datasource>>&& sources,
                   parquet_reader_options const& options,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
  : _stream{stream},
    _mr{mr},
    _options{options.get_timestamp_type(),
             options.get_skip_rows(),
             options.get_num_rows(),
             options.get_row_groups()},
    _sources{std::move(sources)},
    _output_chunk_read_limit{chunk_read_limit},
    _input_pass_read_limit{pass_read_limit}
{
  // Open and parse the source dataset metadata
  _metadata =
    std::make_unique<aggregate_reader_metadata>(_sources, options.is_enabled_use_arrow_schema());

  // Strings may be returned as either string or categorical columns
  _strings_to_categorical = options.is_enabled_convert_strings_to_categories();

  // Binary columns can be read as binary or strings
  _reader_column_schema = options.get_column_schema();

  // Select only columns required by the options and filter
  std::optional<std::vector<std::string>> filter_columns_names;
  if (options.get_filter().has_value() and options.get_columns().has_value()) {
    // list, struct, dictionary are not supported by AST filter yet.
    // extract columns not present in get_columns() & keep count to remove at end.
    filter_columns_names =
      get_column_names_in_expression(options.get_filter(), *(options.get_columns()));
    _num_filter_only_columns = filter_columns_names->size();
  }
  std::tie(_input_columns, _output_buffers, _output_column_schemas) =
    _metadata->select_columns(options.get_columns(),
                              filter_columns_names,
                              options.is_enabled_use_pandas_metadata(),
                              _strings_to_categorical,
                              _options.timestamp_type.id());

  // Save the states of the output buffers for reuse in `chunk_read()`.
  for (auto const& buff : _output_buffers) {
    _output_buffers_template.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
  }

  // Save the name to reference converter to extract output filter AST in
  // `preprocess_file()` and `finalize_output()`
  table_metadata metadata;
  populate_metadata(metadata);
  _expr_conv = named_to_reference_converter(options.get_filter(), metadata);
}

void reader::impl::prepare_data(read_mode mode)
{
  // if we have not preprocessed at the whole-file level, do that now
  if (!_file_preprocessed) {
    // setup file level information
    // - read row group information
    // - setup information on (parquet) chunks
    // - compute schedule of input passes
    preprocess_file(mode);
  }

  // handle any chunking work (ratcheting through the subpasses and chunks within
  // our current pass) if in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) { handle_chunking(mode); }
}

void reader::impl::populate_metadata(table_metadata& out_metadata)
{
  // Return column names
  out_metadata.schema_info.resize(_output_buffers.size());
  for (size_t i = 0; i < _output_column_schemas.size(); i++) {
    auto const& schema                      = _metadata->get_schema(_output_column_schemas[i]);
    out_metadata.schema_info[i].name        = schema.name;
    out_metadata.schema_info[i].is_nullable = schema.repetition_type != REQUIRED;
  }

  // Return user metadata
  out_metadata.per_file_user_data = _metadata->get_key_value_metadata();
  out_metadata.user_data          = {out_metadata.per_file_user_data[0].begin(),
                                     out_metadata.per_file_user_data[0].end()};
}

table_with_metadata reader::impl::read_chunk_internal(read_mode mode)
{
  // If `_output_metadata` has been constructed, just copy it over.
  auto out_metadata = _output_metadata ? table_metadata{*_output_metadata} : table_metadata{};
  out_metadata.schema_info.resize(_output_buffers.size());

  // output cudf columns as determined by the top level schema
  auto out_columns = std::vector<std::unique_ptr<column>>{};
  out_columns.reserve(_output_buffers.size());

  // no work to do (this can happen on the first pass if we have no rows to read)
  if (!has_more_work()) { return finalize_output(out_metadata, out_columns); }

  auto& pass            = *_pass_itm_data;
  auto& subpass         = *pass.subpass;
  auto const& read_info = subpass.output_chunk_read_info[subpass.current_output_chunk];

  // Allocate memory buffers for the output columns.
  allocate_columns(mode, read_info.skip_rows, read_info.num_rows);

  // Parse data into the output buffers.
  decode_page_data(mode, read_info.skip_rows, read_info.num_rows);

  // Create the final output cudf columns.
  for (size_t i = 0; i < _output_buffers.size(); ++i) {
    auto metadata           = _reader_column_schema.has_value()
                                ? std::make_optional<reader_column_schema>((*_reader_column_schema)[i])
                                : std::nullopt;
    auto const& schema      = _metadata->get_schema(_output_column_schemas[i]);
    auto const logical_type = schema.logical_type.value_or(LogicalType{});
    // FIXED_LEN_BYTE_ARRAY never read as string.
    // TODO: if we ever decide that the default reader behavior is to treat unannotated BINARY as
    // binary and not strings, this test needs to change.
    if (schema.type == FIXED_LEN_BYTE_ARRAY and logical_type.type != LogicalType::DECIMAL) {
      metadata = std::make_optional<reader_column_schema>();
      metadata->set_convert_binary_to_strings(false);
      metadata->set_type_length(schema.type_length);
    }
    // Only construct `out_metadata` if `_output_metadata` has not been cached.
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(make_column(_output_buffers[i], &col_name, metadata, _stream));
    } else {
      out_columns.emplace_back(make_column(_output_buffers[i], nullptr, metadata, _stream));
    }
  }

  // Add empty columns if needed. Filter output columns based on filter.
  return finalize_output(out_metadata, out_columns);
}

table_with_metadata reader::impl::finalize_output(table_metadata& out_metadata,
                                                  std::vector<std::unique_ptr<column>>& out_columns)
{
  // Create empty columns as needed (this can happen if we've ended up with no actual data to read)
  for (size_t i = out_columns.size(); i < _output_buffers.size(); ++i) {
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], &col_name, _stream, _mr));
    } else {
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], nullptr, _stream, _mr));
    }
  }

  if (!_output_metadata) {
    populate_metadata(out_metadata);
    // Finally, save the output table metadata into `_output_metadata` for reuse next time.
    _output_metadata = std::make_unique<table_metadata>(out_metadata);
  }

  // advance output chunk/subpass/pass info for non-empty tables if and only if we are in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) {
    auto& pass    = *_pass_itm_data;
    auto& subpass = *pass.subpass;
    subpass.current_output_chunk++;
  }

  // increment the output chunk count
  _file_itm_data._output_chunk_count++;

  // check if the output filter AST expression (= _expr_conv.get_converted_expr()) exists
  if (_expr_conv.get_converted_expr().has_value()) {
    auto read_table = std::make_unique<table>(std::move(out_columns));
    auto predicate  = cudf::detail::compute_column(*read_table,
                                                  _expr_conv.get_converted_expr().value().get(),
                                                  _stream,
                                                  rmm::mr::get_current_device_resource());
    CUDF_EXPECTS(predicate->view().type().id() == type_id::BOOL8,
                 "Predicate filter should return a boolean");
    // Exclude columns present in filter only in output
    auto counting_it        = thrust::make_counting_iterator<std::size_t>(0);
    auto const output_count = read_table->num_columns() - _num_filter_only_columns;
    auto only_output        = read_table->select(counting_it, counting_it + output_count);
    auto output_table = cudf::detail::apply_boolean_mask(only_output, *predicate, _stream, _mr);
    if (_num_filter_only_columns > 0) { out_metadata.schema_info.resize(output_count); }
    return {std::move(output_table), std::move(out_metadata)};
  }
  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

table_with_metadata reader::impl::read()
{
  CUDF_EXPECTS(_output_chunk_read_limit == 0,
               "Reading the whole file must not have non-zero byte_limit.");

  prepare_data(read_mode::READ_ALL);
  return read_chunk_internal(read_mode::READ_ALL);
}

table_with_metadata reader::impl::read_chunk()
{
  // Reset the output buffers to their original states (right after reader construction).
  // Don't need to do it if we read the file all at once.
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes() and
      not is_first_output_chunk()) {
    _output_buffers.resize(0);
    for (auto const& buff : _output_buffers_template) {
      _output_buffers.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }

  prepare_data(read_mode::CHUNKED_READ);
  return read_chunk_internal(read_mode::CHUNKED_READ);
}

bool reader::impl::has_next()
{
  prepare_data(read_mode::CHUNKED_READ);

  // current_input_pass will only be incremented to be == num_passes after
  // the last chunk in the last subpass in the last pass has been returned
  // if not has_more_work then check if this is the first pass in an empty
  // table and return true so it could be read once.
  return has_more_work() or is_first_output_chunk();
}

namespace {
parquet_column_schema walk_schema(aggregate_reader_metadata const* mt, int idx)
{
  SchemaElement const& sch = mt->get_schema(idx);
  std::vector<parquet_column_schema> children;
  for (auto const& child_idx : sch.children_idx) {
    children.push_back(walk_schema(mt, child_idx));
  }
  return parquet_column_schema{
    sch.name, static_cast<parquet::TypeKind>(sch.type), std::move(children)};
}
}  // namespace

parquet_metadata read_parquet_metadata(host_span<std::unique_ptr<datasource> const> sources)
{
  // do not use arrow schema when reading information from parquet metadata.
  static constexpr auto use_arrow_schema = false;

  // Open and parse the source dataset metadata
  auto metadata = aggregate_reader_metadata(sources, use_arrow_schema);

  return parquet_metadata{parquet_schema{walk_schema(&metadata, 0)},
                          metadata.get_num_rows(),
                          metadata.get_num_row_groups(),
                          metadata.get_key_value_metadata()[0],
                          metadata.get_rowgroup_metadata()};
}

}  // namespace cudf::io::parquet::detail
