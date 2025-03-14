/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "hybrid_scan_impl.hpp"

#include "cudf/io/text/byte_range_info.hpp"
#include "hybrid_scan_helpers.hpp"

#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <bitset>
#include <iterator>
#include <limits>
#include <numeric>

namespace cudf::experimental::io::parquet::detail {

namespace {
// Tests the passed in logical type for a FIXED_LENGTH_BYTE_ARRAY column to see if it should
// be treated as a string. Currently the only logical type that has special handling is DECIMAL.
// Other valid types in the future would be UUID (still treated as string) and FLOAT16 (which
// for now would also be treated as a string).
[[maybe_unused]] inline bool is_treat_fixed_length_as_string(
  std::optional<cudf::io::parquet::detail::LogicalType> const& logical_type)
{
  if (!logical_type.has_value()) { return true; }
  return logical_type->type != cudf::io::parquet::detail::LogicalType::DECIMAL;
}

}  // namespace

void impl::decode_page_data(size_t skip_rows, size_t num_rows)
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

  size_t const sum_max_depths =
    std::accumulate(pass.chunks.begin(),
                    pass.chunks.end(),
                    0,
                    [&](size_t cursum, cudf::io::parquet::detail::ColumnChunkDesc const& chunk) {
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
  auto const has_strings = (kernel_mask & cudf::io::parquet::detail::STRINGS_MASK) != 0;
  auto col_string_sizes  = cudf::detail::make_host_vector<size_t>(_input_columns.size(), _stream);
  if (has_strings) {
    // need to compute pages bounds/sizes if we lack page indexes or are using custom bounds
    // TODO: we could probably dummy up size stats for FLBA data since we know the width
    auto const has_flba =
      std::any_of(pass.chunks.begin(), pass.chunks.end(), [](auto const& chunk) {
        return chunk.physical_type == cudf::io::parquet::detail::FIXED_LEN_BYTE_ARRAY and
               is_treat_fixed_length_as_string(chunk.logical_type);
      });

    if (!_has_page_index || _uses_custom_row_bounds || has_flba) {
      ComputePageStringSizes(subpass.pages,
                             pass.chunks,
                             delta_temp_buf,
                             skip_rows,
                             num_rows,
                             level_type_size,
                             kernel_mask,
                             _stream);
    }

    // Compute column string sizes (using page string offsets) for this output table chunk
    col_string_sizes = calculate_page_string_offsets();

    // Check for overflow in cumulative column string sizes of this pass so that the page string
    // offsets of overflowing (large) string columns are treated as 64-bit.
    auto const threshold         = static_cast<size_t>(strings::detail::get_offset64_threshold());
    auto const has_large_strings = std::any_of(col_string_sizes.cbegin(),
                                               col_string_sizes.cend(),
                                               [=](std::size_t sz) { return sz > threshold; });
    if (has_large_strings and not strings::detail::is_large_strings_enabled()) {
      CUDF_FAIL("String column exceeds the column size limit", std::overflow_error);
    }

    // Mark/unmark column-chunk descriptors depending on the string sizes of corresponding output
    // column chunks and the large strings threshold.
    for (auto& chunk : pass.chunks) {
      auto const idx            = chunk.src_col_index;
      chunk.is_large_string_col = (col_string_sizes[idx] > threshold);
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

    size_t const max_depth = _metadata->get_output_nesting_depth(pass.chunks[c].src_col_schema);
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

    auto* cols = &_output_buffers;
    for (size_t idx = 0; idx < max_depth; idx++) {
      auto& out_buf = (*cols)[input_col.nesting[idx]];
      cols          = &out_buf.children;

      int const owning_schema =
        out_buf.user_data & cudf::io::parquet::detail::PARQUET_COLUMN_BUFFER_SCHEMA_MASK;
      if (owning_schema == 0 || owning_schema == input_col.schema_idx) {
        valids[idx] = out_buf.null_mask();
        data[idx]   = out_buf.data();
        // only do string buffer for leaf
        if (idx == max_depth - 1 and out_buf.string_size() == 0 and
            col_string_sizes[pass.chunks[c].src_col_index] > 0) {
          out_buf.create_string_data(col_string_sizes[pass.chunks[c].src_col_index],
                                     pass.chunks[c].is_large_string_col,
                                     _stream);
        }
        if (has_strings) { str_data[idx] = out_buf.string_data(); }
        out_buf.user_data |= static_cast<uint32_t>(input_col.schema_idx) &
                             cudf::io::parquet::detail::PARQUET_COLUMN_BUFFER_SCHEMA_MASK;
      } else {
        valids[idx] = nullptr;
        data[idx]   = nullptr;
      }
    }
  }

  // Create an empty device vector to store the initial str offset for large string columns from for
  // string decoders.
  auto initial_str_offsets = rmm::device_uvector<size_t>{0, _stream, _mr};

  pass.chunks.host_to_device_async(_stream);
  chunk_nested_valids.host_to_device_async(_stream);
  chunk_nested_data.host_to_device_async(_stream);
  if (has_strings) {
    // Host vector to initialize the initial string offsets
    auto host_offsets_vector =
      cudf::detail::make_host_vector<size_t>(_input_columns.size(), _stream);
    std::fill(
      host_offsets_vector.begin(), host_offsets_vector.end(), std::numeric_limits<size_t>::max());
    // Initialize the initial string offsets vector from the host vector
    initial_str_offsets =
      cudf::detail::make_device_uvector_async(host_offsets_vector, _stream, _mr);
    chunk_nested_str_data.host_to_device_async(_stream);
  }

  // create this before we fork streams
  cudf::io::parquet::kernel_error error_code(_stream);

  // get the number of streams we need from the pool and tell them to wait on the H2D copies
  int const nkernels = std::bitset<32>(kernel_mask).count();
  auto streams       = cudf::detail::fork_streams(_stream, nkernels);

  int s_idx = 0;

  auto decode_data = [&](cudf::io::parquet::detail::decode_kernel_mask decoder_mask) {
    DecodePageData(subpass.pages,
                   pass.chunks,
                   num_rows,
                   skip_rows,
                   level_type_size,
                   decoder_mask,
                   initial_str_offsets,
                   error_code.data(),
                   streams[s_idx++]);
  };

  using decode_kernel_mask = cudf::io::parquet::detail::decode_kernel_mask;

  // launch string decoder for plain encoded flat columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING) != 0) {
    decode_data(decode_kernel_mask::STRING);
  }

  // launch string decoder for plain encoded nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_NESTED) != 0) {
    decode_data(decode_kernel_mask::STRING_NESTED);
  }

  // launch string decoder for plain encoded list columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_LIST) != 0) {
    decode_data(decode_kernel_mask::STRING_LIST);
  }

  // launch string decoder for dictionary encoded flat columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_DICT) != 0) {
    decode_data(decode_kernel_mask::STRING_DICT);
  }

  // launch string decoder for dictionary encoded nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_DICT_NESTED) != 0) {
    decode_data(decode_kernel_mask::STRING_DICT_NESTED);
  }

  // launch string decoder for dictionary encoded list columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_DICT_LIST) != 0) {
    decode_data(decode_kernel_mask::STRING_DICT_LIST);
  }

  // launch string decoder for byte-stream-split encoded flat columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_STREAM_SPLIT) != 0) {
    decode_data(decode_kernel_mask::STRING_STREAM_SPLIT);
  }

  // launch string decoder for byte-stream-split encoded nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_STREAM_SPLIT_NESTED) != 0) {
    decode_data(decode_kernel_mask::STRING_STREAM_SPLIT_NESTED);
  }

  // launch string decoder for byte-stream-split encoded list columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_STREAM_SPLIT_LIST) != 0) {
    decode_data(decode_kernel_mask::STRING_STREAM_SPLIT_LIST);
  }

  // launch delta byte array decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_BYTE_ARRAY) != 0) {
    DecodeDeltaByteArray(subpass.pages,
                         pass.chunks,
                         num_rows,
                         skip_rows,
                         level_type_size,
                         initial_str_offsets,
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
                               initial_str_offsets,
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
  if (BitAnd(kernel_mask, decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT) != 0) {
    decode_data(decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT);
  }

  // launch byte stream split decoder, for nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED) != 0) {
    decode_data(decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED);
  }

  // launch byte stream split decoder, for list columns
  if (BitAnd(kernel_mask, decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST) != 0) {
    decode_data(decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST);
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

  // launch fixed width type decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_NO_DICT) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_NO_DICT);
  }

  // launch fixed width type decoder for lists
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST);
  }

  // launch fixed width type decoder, for nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED);
  }

  // launch boolean type decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::BOOLEAN) != 0) {
    decode_data(decode_kernel_mask::BOOLEAN);
  }

  // launch boolean type decoder, for nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::BOOLEAN_NESTED) != 0) {
    decode_data(decode_kernel_mask::BOOLEAN_NESTED);
  }

  // launch boolean type decoder, for nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::BOOLEAN_LIST) != 0) {
    decode_data(decode_kernel_mask::BOOLEAN_LIST);
  }

  // launch fixed width type decoder with dictionaries
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_DICT) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_DICT);
  }

  // launch fixed width type decoder with dictionaries for lists
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_DICT_LIST) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_DICT_LIST);
  }

  // launch fixed width type decoder with dictionaries, for nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_DICT_NESTED) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_DICT_NESTED);
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

  // Copy over initial string offsets from device
  auto h_initial_str_offsets = cudf::detail::make_host_vector_async(initial_str_offsets, _stream);

  if (auto const error = error_code.value_sync(_stream); error != 0) {
    CUDF_FAIL("Parquet data decode failed with code(s) " +
              cudf::io::parquet::kernel_error::to_string(error));
  }

  // For list and string columns, add the final offset to every offset buffer.
  // Note : the reason we are doing this here instead of in the decode kernel is
  // that it is difficult/impossible for a given page to know that it is writing the very
  // last value that should then be followed by a terminator (because rows can span
  // page boundaries).
  std::vector<size_type*> out_buffers;
  std::vector<size_type> final_offsets;
  out_buffers.reserve(_input_columns.size());
  final_offsets.reserve(_input_columns.size());
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    input_column_info const& input_col = _input_columns[idx];

    auto* cols = &_output_buffers;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      if (out_buf.type.id() == type_id::LIST &&
          (out_buf.user_data &
           cudf::io::parquet::detail::PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED) == 0) {
        CUDF_EXPECTS(l_idx < input_col.nesting_depth() - 1, "Encountered a leaf list column");
        auto const& child = (*cols)[input_col.nesting[l_idx + 1]];

        // the final offset for a list at level N is the size of it's child
        size_type const offset = child.type.id() == type_id::LIST ? child.size - 1 : child.size;
        out_buffers.emplace_back(static_cast<size_type*>(out_buf.data()) + (out_buf.size - 1));
        final_offsets.emplace_back(offset);
        out_buf.user_data |= cudf::io::parquet::detail::PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED;
      } else if (out_buf.type.id() == type_id::STRING) {
        // only if it is not a large strings column
        if (col_string_sizes[idx] <=
            static_cast<size_t>(strings::detail::get_offset64_threshold())) {
          out_buffers.emplace_back(static_cast<size_type*>(out_buf.data()) + out_buf.size);
          final_offsets.emplace_back(static_cast<size_type>(col_string_sizes[idx]));
        }
        // Nested large strings column
        else if (input_col.nesting_depth() > 0) {
          CUDF_EXPECTS(h_initial_str_offsets[idx] != std::numeric_limits<size_t>::max(),
                       "Encountered invalid initial offset for large string column");
          out_buf.set_initial_string_offset(h_initial_str_offsets[idx]);
        }
      }
    }
  }
  // Write the final offsets for list and string columns in a batched manner
  cudf::io::parquet::detail::WriteFinalOffsets(final_offsets, out_buffers, _stream);

  // update null counts in the final column buffers
  for (size_t idx = 0; idx < subpass.pages.size(); idx++) {
    cudf::io::parquet::detail::PageInfo* pi = &subpass.pages[idx];
    if (pi->flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY) { continue; }
    cudf::io::parquet::detail::ColumnChunkDesc* col = &pass.chunks[pi->chunk_idx];
    input_column_info const& input_col              = _input_columns[col->src_col_index];

    int const index = pi->nesting_decode - page_nesting_decode.device_ptr();
    cudf::io::parquet::detail::PageNestingDecodeInfo* pndi = &page_nesting_decode[index];

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

impl::impl(cudf::host_span<uint8_t const> footer_bytes,
           cudf::host_span<uint8_t const> page_index_bytes,
           cudf::io::parquet_reader_options const& options)
{
  // Open and parse the source dataset metadata
  _metadata = std::make_unique<aggregate_reader_metadata>(
    footer_bytes,
    page_index_bytes,
    options.is_enabled_use_arrow_schema(),
    options.get_columns().has_value() and options.is_enabled_allow_mismatched_pq_schemas());

  // Strings may be returned as either string or categorical columns
  auto const strings_to_categorical = options.is_enabled_convert_strings_to_categories();

  // Select only columns required by the filter
  std::optional<std::vector<std::string>> filter_columns_names;
  if (options.get_filter().has_value() and options.get_columns().has_value()) {
    // list, struct, dictionary are not supported by AST filter yet.
    // extract columns not present in get_columns() & keep count to remove at end.
    filter_columns_names = cudf::io::parquet::detail::get_column_names_in_expression(
      options.get_filter(), *(options.get_columns()));
    _num_filter_only_columns = filter_columns_names->size();
  }

  // Only need to select columns if filter is available
  if (options.get_filter().has_value()) {
    if (not filter_columns_names.has_value()) {
      filter_columns_names =
        cudf::io::parquet::detail::get_column_names_in_expression(options.get_filter(), {});
    }

    std::tie(_input_columns, _output_buffers, _output_column_schemas) =
      _metadata->select_filter_columns(filter_columns_names,
                                       options.is_enabled_use_pandas_metadata(),
                                       strings_to_categorical,
                                       options.get_timestamp_type().id());

    // Save the states of the output buffers for reuse.
    for (auto const& buff : _output_buffers) {
      _output_buffers_template.emplace_back(
        cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }
}

std::vector<size_type> impl::get_valid_row_groups(
  cudf::io::parquet_reader_options const& options) const
{
  auto const num_row_groups = _metadata->get_num_row_groups();
  auto row_groups_indices   = std::vector<size_type>(num_row_groups);
  std::iota(row_groups_indices.begin(), row_groups_indices.end(), size_type{0});
  return row_groups_indices;
}

std::vector<std::vector<size_type>> impl::filter_row_groups_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  return _metadata->filter_row_groups_with_stats(row_group_indices,
                                                 output_dtypes,
                                                 _output_column_schemas,
                                                 expr_conv.get_converted_expr(),
                                                 stream);
}

std::pair<std::vector<cudf::io::text::byte_range_info>,
          std::vector<cudf::io::text::byte_range_info>>
impl::get_secondary_filters(cudf::host_span<std::vector<size_type> const> row_group_indices,
                            cudf::io::parquet_reader_options const& options) const
{
  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  auto const bloom_filter_bytes = _metadata->get_bloom_filter_bytes(
    row_group_indices, output_dtypes, _output_column_schemas, expr_conv.get_converted_expr());
  auto const dictionary_page_bytes = _metadata->get_dictionary_page_bytes(
    row_group_indices, output_dtypes, _output_column_schemas, expr_conv.get_converted_expr());

  return {bloom_filter_bytes, dictionary_page_bytes};
}

std::vector<std::vector<size_type>> impl::filter_row_groups_with_dictionary_pages(
  std::vector<rmm::device_buffer>& dictionary_page_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  return _metadata->filter_row_groups_with_dictionary_pages(dictionary_page_data,
                                                            row_group_indices,
                                                            output_dtypes,
                                                            _output_column_schemas,
                                                            expr_conv.get_converted_expr(),
                                                            stream);
}

std::vector<std::vector<size_type>> impl::filter_row_groups_with_bloom_filters(
  std::vector<rmm::device_buffer>& bloom_filter_data,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream) const
{
  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  return _metadata->filter_row_groups_with_bloom_filters(bloom_filter_data,
                                                         row_group_indices,
                                                         output_dtypes,
                                                         _output_column_schemas,
                                                         expr_conv.get_converted_expr(),
                                                         stream);
}

std::vector<std::vector<bool>> impl::filter_data_pages_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  if (not _file_preprocessed) { prepare_row_groups(row_group_indices, options); }

  auto mr = cudf::get_current_device_resource_ref();

  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  auto predicate = _metadata->filter_data_pages_with_stats(row_group_indices,
                                                           output_dtypes,
                                                           _output_column_schemas,
                                                           expr_conv.get_converted_expr(),
                                                           stream,
                                                           mr);

  return _metadata->compute_data_page_validity(
    predicate->view(), row_group_indices, output_dtypes, _output_column_schemas, stream);
}

std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>>
impl::get_column_chunk_byte_ranges(cudf::host_span<std::vector<size_type> const> row_group_indices,
                                   cudf::io::parquet_reader_options const& options) const
{
  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_row_groups =
    std::accumulate(row_group_indices.begin(),
                    row_group_indices.end(),
                    size_t{0},
                    [](size_t sum, auto const& row_groups) { return sum + row_groups.size(); });
  auto const num_chunks = num_row_groups * num_input_columns;

  // Association between each column chunk and its source
  std::vector<size_type> chunk_source_map(num_chunks);

  // Keep track of column chunk byte ranges
  std::vector<cudf::io::text::byte_range_info> column_chunk_byte_ranges(num_chunks);

  size_type chunk_count = 0;
  for (size_t src_idx = 0; src_idx < row_group_indices.size(); ++src_idx) {
    auto const& row_groups = row_group_indices[src_idx];
    for (auto const row_group_index : row_groups) {
      // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
      for (size_t i = 0; i < num_input_columns; ++i) {
        auto const& col = _input_columns[i];
        // look up metadata
        auto& col_meta = _metadata->get_column_metadata(row_group_index, src_idx, col.schema_idx);
        auto const chunk_offset =
          (col_meta.dictionary_page_offset != 0)
            ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
            : col_meta.data_page_offset;
        auto const chunk_size = col_meta.total_compressed_size;

        column_chunk_byte_ranges[chunk_count] = {chunk_offset, chunk_size};

        // Map each column chunk to its column index and its source index
        chunk_source_map[chunk_count] = src_idx;

        chunk_count++;
      }
    }
  }

  return {std::move(column_chunk_byte_ranges), std::move(chunk_source_map)};
}

cudf::io::table_with_metadata impl::materialize_filter_columns(
  cudf::host_span<std::vector<bool> const> filtered_data_pages,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  initialize_options(row_group_indices, filtered_data_pages, options, stream);
  prepare_data(row_group_indices, std::move(column_chunk_buffers), options);

  // Make sure we haven't gone past the input passes
  CUDF_EXPECTS(_file_itm_data._current_input_pass < _file_itm_data.num_passes(), "");
  return read_chunk_internal();
}

void impl::initialize_options(cudf::host_span<std::vector<size_type> const> row_group_indices,
                              cudf::host_span<std::vector<bool> const> filtered_data_pages,
                              cudf::io::parquet_reader_options const& options,
                              rmm::cuda_stream_view stream)
{
  // Save the name to reference converter to extract output filter AST in
  // `preprocess_file()` and `finalize_output()`
  table_metadata metadata;
  populate_metadata(metadata);
  _expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  _data_page_validity = filtered_data_pages;

  _uses_custom_row_bounds = (options.get_num_rows().has_value() or options.get_skip_rows() > 0);

  // Strings may be returned as either string or categorical columns
  _strings_to_categorical = options.is_enabled_convert_strings_to_categories();

  // Binary columns can be read as binary or strings
  _reader_column_schema = options.get_column_schema();

  _timestamp_type = options.get_timestamp_type();

  _num_sources = row_group_indices.size();

  // CUDA stream to use for internal operations
  _stream = stream;
}

cudf::io::table_with_metadata impl::read_chunk_internal()
{
  // If `_output_metadata` has been constructed, just copy it over.
  auto out_metadata = _output_metadata ? table_metadata{*_output_metadata} : table_metadata{};
  out_metadata.schema_info.resize(_output_buffers.size());

  // output cudf columns as determined by the top level schema
  auto out_columns = std::vector<std::unique_ptr<column>>{};
  out_columns.reserve(_output_buffers.size());

  // Copy number of total input row groups and number of surviving row groups from predicate
  // pushdown.
  out_metadata.num_input_row_groups = _file_itm_data.num_input_row_groups;
  // Copy the number surviving row groups from each predicate pushdown only if the filter has
  // value.
  if (_expr_conv.get_converted_expr().has_value()) {
    out_metadata.num_row_groups_after_stats_filter =
      _file_itm_data.surviving_row_groups.after_stats_filter;
    out_metadata.num_row_groups_after_bloom_filter =
      _file_itm_data.surviving_row_groups.after_bloom_filter;
  }

  // no work to do (this can happen on the first pass if we have no rows to read)
  if (!has_more_work()) {
    // Finalize output
    return finalize_output(out_metadata, out_columns);
  }

  auto& pass            = *_pass_itm_data;
  auto& subpass         = *pass.subpass;
  auto const& read_info = subpass.output_chunk_read_info[subpass.current_output_chunk];

  // Allocate memory buffers for the output columns.
  allocate_columns(read_info.skip_rows, read_info.num_rows);

  // Parse data into the output buffers.
  decode_page_data(read_info.skip_rows, read_info.num_rows);

  // Create the final output cudf columns.
  for (size_t i = 0; i < _output_buffers.size(); ++i) {
    auto metadata      = _reader_column_schema.has_value()
                           ? std::make_optional<reader_column_schema>((*_reader_column_schema)[i])
                           : std::nullopt;
    auto const& schema = _metadata->get_schema(_output_column_schemas[i]);
    auto const logical_type =
      schema.logical_type.value_or(cudf::io::parquet::detail::LogicalType{});
    // FIXED_LEN_BYTE_ARRAY never read as string.
    // TODO: if we ever decide that the default reader behavior is to treat unannotated BINARY
    // as binary and not strings, this test needs to change.
    if (schema.type == cudf::io::parquet::detail::FIXED_LEN_BYTE_ARRAY and
        logical_type.type != cudf::io::parquet::detail::LogicalType::DECIMAL) {
      metadata = std::make_optional<reader_column_schema>();
      metadata->set_convert_binary_to_strings(false);
      metadata->set_type_length(schema.type_length);
    }
    // Only construct `out_metadata` if `_output_metadata` has not been cached.
    if (!_output_metadata) {
      cudf::io::column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(make_column(_output_buffers[i], &col_name, metadata, _stream));
    } else {
      out_columns.emplace_back(make_column(_output_buffers[i], nullptr, metadata, _stream));
    }
  }

  // Add empty columns if needed. Filter output columns based on filter.
  return finalize_output(out_metadata, out_columns);
}

cudf::io::table_with_metadata impl::finalize_output(
  table_metadata& out_metadata, std::vector<std::unique_ptr<column>>& out_columns)
{
  // Create empty columns as needed (this can happen if we've ended up with no actual data to
  // read)
  for (size_t i = out_columns.size(); i < _output_buffers.size(); ++i) {
    if (!_output_metadata) {
      cudf::io::column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(
        cudf::io::detail::empty_like(_output_buffers[i], &col_name, _stream, _mr));
    } else {
      out_columns.emplace_back(
        cudf::io::detail::empty_like(_output_buffers[i], nullptr, _stream, _mr));
    }
  }

  if (!_output_metadata) {
    populate_metadata(out_metadata);
    // Finally, save the output table metadata into `_output_metadata` for reuse next time.
    _output_metadata = std::make_unique<cudf::io::table_metadata>(out_metadata);
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
                                                  cudf::get_current_device_resource_ref());
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

void impl::populate_metadata(table_metadata& out_metadata) const
{
  // Return column names
  out_metadata.schema_info.resize(_output_buffers.size());
  for (size_t i = 0; i < _output_column_schemas.size(); i++) {
    auto const& schema               = _metadata->get_schema(_output_column_schemas[i]);
    out_metadata.schema_info[i].name = schema.name;
    out_metadata.schema_info[i].is_nullable =
      schema.repetition_type != cudf::io::parquet::detail::REQUIRED;
  }

  // Return user metadata
  out_metadata.per_file_user_data = _metadata->get_key_value_metadata();
  out_metadata.user_data          = {out_metadata.per_file_user_data[0].begin(),
                                     out_metadata.per_file_user_data[0].end()};
}

void impl::prepare_data(cudf::host_span<std::vector<size_type> const> row_group_indices,
                        std::vector<rmm::device_buffer> column_chunk_buffers,
                        cudf::io::parquet_reader_options const& options)
{
  // if we have not preprocessed at the whole-file level, do that now
  if (not _file_preprocessed) {
    // setup file level information
    // - read row group information
    // - setup information on (parquet) chunks
    // - compute schedule of input passes
    prepare_row_groups(row_group_indices, options);
  }

  // handle any chunking work (ratcheting through the subpasses and chunks within
  // our current pass) if in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) {
    handle_chunking(std::move(column_chunk_buffers), options);
  }
}

}  // namespace cudf::experimental::io::parquet::detail
