/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_helpers.hpp"
#include "hybrid_scan_impl.hpp"
#include "io/parquet/reader_impl_preprocess_utils.cuh"
#include "io/utilities/time_utils.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/batched_memset.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

#include <numeric>

namespace cudf::io::parquet::experimental::detail {

namespace {

using parquet::detail::chunk_page_info;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::PageInfo;

/**
 * @brief Decode the dictionary page information from each column chunk
 *
 * @param chunks Host device span of column chunk descriptors, one per input column chunk
 * @param pages Host device span of empty page headers to fill in, one per input column chunk
 * @param stream CUDA stream
 */
void decode_dictionary_page_headers(cudf::detail::hostdevice_span<ColumnChunkDesc> chunks,
                                    cudf::detail::hostdevice_span<PageInfo> pages,
                                    rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  std::vector<size_t> host_chunk_page_counts(chunks.size() + 1);
  std::transform(
    chunks.host_begin(), chunks.host_end(), host_chunk_page_counts.begin(), [](auto const& chunk) {
      return chunk.num_dict_pages;
    });
  host_chunk_page_counts[chunks.size()] = 0;

  auto chunk_page_counts = cudf::detail::make_device_uvector_async(
    host_chunk_page_counts, stream, cudf::get_current_device_resource_ref());

  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         chunk_page_counts.begin(),
                         chunk_page_counts.end(),
                         chunk_page_counts.begin(),
                         size_t{0},
                         cuda::std::plus<size_t>{});

  rmm::device_uvector<chunk_page_info> d_chunk_page_info(chunks.size(), stream);

  thrust::for_each(rmm::exec_policy_nosync(stream),
                   thrust::counting_iterator<cuda::std::size_t>(0),
                   thrust::counting_iterator(chunks.size()),
                   [cpi               = d_chunk_page_info.begin(),
                    chunk_page_counts = chunk_page_counts.begin(),
                    pages             = pages.device_begin()] __device__(size_t i) {
                     cpi[i].pages = &pages[chunk_page_counts[i]];
                   });

  parquet::kernel_error error_code(stream);

  parquet::detail::decode_page_headers(
    chunks.device_begin(), d_chunk_page_info.begin(), chunks.size(), error_code.data(), stream);

  if (auto const error = error_code.value_sync(stream); error != 0) {
    CUDF_FAIL("Parquet header parsing failed with code(s) " +
              parquet::kernel_error::to_string(error));
  }

  // Setup dictionary page for each chunk
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   pages.device_begin(),
                   pages.device_end(),
                   [chunks = chunks.device_begin()] __device__(PageInfo const& p) {
                     if (p.flags & parquet::detail::PAGEINFO_FLAGS_DICTIONARY) {
                       chunks[p.chunk_idx].dict_page = &p;
                     }
                   });

  pages.device_to_host_async(stream);
  chunks.device_to_host_async(stream);
  stream.synchronize();
}

}  // namespace

void hybrid_scan_reader_impl::prepare_row_groups(
  read_mode mode, cudf::host_span<std::vector<size_type> const> row_group_indices)
{
  std::tie(_file_itm_data.global_skip_rows,
           _file_itm_data.global_num_rows,
           _file_itm_data.row_groups,
           _file_itm_data.num_rows_per_source,
           _file_itm_data.num_input_row_groups,
           _file_itm_data.surviving_row_groups) =
    _extended_metadata->select_row_groups(
      {}, row_group_indices, {}, {}, {}, {}, {}, {}, {}, _stream);

  CUDF_EXPECTS(
    std::cmp_less_equal(_file_itm_data.global_num_rows, std::numeric_limits<size_type>::max()),
    "READ_ALL mode does not support reading number of rows more than cudf's column size limit. "
    "For reading larger number of rows, please use chunked_parquet_reader.",
    std::overflow_error);

  // check for page indexes
  _has_page_index = std::all_of(_file_itm_data.row_groups.cbegin(),
                                _file_itm_data.row_groups.cend(),
                                [](auto const& row_group) { return row_group.has_page_index(); });

  if (_file_itm_data.global_num_rows > 0 && not _file_itm_data.row_groups.empty() &&
      not _input_columns.empty()) {
    // fills in chunk information without physically loading or decompressing
    // the associated data
    create_global_chunk_info();

    // compute schedule of input reads.
    compute_input_passes(mode);
  }

  _file_preprocessed = true;
}

bool hybrid_scan_reader_impl::setup_column_chunks()
{
  auto const& row_groups_info = _pass_itm_data->row_groups;
  auto& chunks                = _pass_itm_data->chunks;

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_chunks        = row_groups_info.size() * num_input_columns;

  // Initialize column chunk information
  size_t total_decompressed_size = 0;
  size_type chunk_count          = 0;
  for (auto const& rg : row_groups_info) {
    // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
    for (size_t i = 0; i < num_input_columns; ++i) {
      auto const& col = _input_columns[i];
      // look up metadata
      auto& col_meta =
        _extended_metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);

      if (col_meta.codec != Compression::UNCOMPRESSED) {
        total_decompressed_size += col_meta.total_uncompressed_size;
      }

      // Set pointer to compressed data
      chunks[chunk_count].compressed_data =
        static_cast<uint8_t const*>(_pass_itm_data->raw_page_data[chunk_count].data());

      chunk_count++;
    }
  }
  return total_decompressed_size > 0;
}

void hybrid_scan_reader_impl::setup_compressed_data(
  std::vector<rmm::device_buffer>&& column_chunk_buffers)
{
  auto& pass = *_pass_itm_data;

  // This function should never be called if `num_rows == 0`.
  CUDF_EXPECTS(_pass_itm_data->num_rows > 0, "Number of reading rows must not be zero.");

  auto& chunks = pass.chunks;

  // Move column chunk buffers to raw page data.
  _pass_itm_data->raw_page_data = std::move(column_chunk_buffers);

  pass.has_compressed_data = setup_column_chunks();

  // Process dataset chunk pages into output columns
  auto const total_pages = _has_page_index ? count_page_headers_with_pgidx(chunks, _stream)
                                           : count_page_headers(chunks, _stream);
  if (total_pages <= 0) { return; }
  rmm::device_uvector<PageInfo> unsorted_pages(total_pages, _stream);

  // decoding of column/page information
  parquet::detail::decode_page_headers(pass, unsorted_pages, _has_page_index, _stream);
  CUDF_EXPECTS(pass.page_offsets.size() - 1 == static_cast<size_t>(_input_columns.size()),
               "Encountered page_offsets / num_columns mismatch");
}

std::tuple<bool,
           cudf::detail::hostdevice_vector<ColumnChunkDesc>,
           cudf::detail::hostdevice_vector<PageInfo>>
hybrid_scan_reader_impl::prepare_dictionaries(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<rmm::device_buffer> dictionary_page_data,
  cudf::host_span<int const> dictionary_col_schemas,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  // Create row group information for the input row group indices
  auto const row_groups_info = std::get<2>(_extended_metadata->select_row_groups(
    {}, row_group_indices, {}, {}, {}, {}, {}, {}, {}, _stream));

  CUDF_EXPECTS(
    row_groups_info.size() * dictionary_col_schemas.size() == dictionary_page_data.size(),
    "Dictionary page data size must match the number of row groups times the number of columns "
    "with dictionaries and an (in)equality predicate");

  // Number of column chunks
  auto const total_column_chunks = dictionary_page_data.size();

  // Boolean to check if any of the column chunnks have compressed data
  auto has_compressed_data = false;

  // Initialize column chunk descriptors
  auto chunks = cudf::detail::hostdevice_vector<cudf::io::parquet::detail::ColumnChunkDesc>(
    total_column_chunks, stream);
  auto chunk_idx = 0;

  // For all row groups
  for (auto const& rg : row_groups_info) {
    auto const& row_group = _extended_metadata->get_row_group(rg.index, rg.source_index);

    // For all columns with dictionary page and (in)equality predicate
    for (auto col_schema_idx : dictionary_col_schemas) {
      // look up metadata
      auto& col_meta =
        _extended_metadata->get_column_metadata(rg.index, rg.source_index, col_schema_idx);
      auto& schema = _extended_metadata->get_schema(
        _extended_metadata->map_schema_index(col_schema_idx, rg.source_index), rg.source_index);

      // dictionary data buffer for this column chunk
      auto& dict_page_data = dictionary_page_data[chunk_idx];

      // Check if the column chunk has compressed data
      has_compressed_data |=
        col_meta.codec != Compression::UNCOMPRESSED and col_meta.total_compressed_size > 0;

      // TODO: Use `parquet::detail::conversion_info` instead of directly computing `clock_rate`
      // when AST support for decimals is available
      auto const column_type_id =
        parquet::detail::to_type_id(schema,
                                    options.is_enabled_convert_strings_to_categories(),
                                    options.get_timestamp_type().id());
      auto const clock_rate = is_chrono(data_type{column_type_id})
                                ? to_clockrate(options.get_timestamp_type().id())
                                : int32_t{0};

      // Create a column chunk descriptor - zero/null values for all fields that are not needed
      chunks[chunk_idx] = ColumnChunkDesc(static_cast<int64_t>(dict_page_data.size()),
                                          static_cast<uint8_t*>(dict_page_data.data()),
                                          col_meta.num_values,
                                          schema.type,
                                          schema.type_length,
                                          0,  // start_row
                                          0,  // num_rows
                                          0,  // max_definition_level
                                          0,  // max_repetition_level
                                          0,  // max_nesting_depth
                                          0,  // def_level_bits
                                          0,  // rep_level_bits
                                          col_meta.codec,
                                          schema.logical_type,
                                          clock_rate,
                                          0,  // src_col_index
                                          col_schema_idx,
                                          nullptr,  // chunk_info
                                          0.0f,     // list_bytes_per_row_est
                                          false,    // strings_to_categorical
                                          rg.source_index);
      // Set the number of dictionary and data pages
      chunks[chunk_idx].num_dict_pages = static_cast<int32_t>(dict_page_data.size() > 0);
      chunks[chunk_idx].num_data_pages = 0;  // Always zero at this stage
      chunk_idx++;
    }
  }

  // Copy the column chunk descriptors to the device
  chunks.host_to_device_async(stream);

  // Create page infos for each column chunk's dictionary page
  cudf::detail::hostdevice_vector<PageInfo> pages(total_column_chunks, stream);

  // Decode dictionary page headers
  decode_dictionary_page_headers(chunks, pages, stream);

  return {has_compressed_data, std::move(chunks), std::move(pages)};
}

void hybrid_scan_reader_impl::update_row_mask(cudf::column_view const& in_row_mask,
                                              cudf::mutable_column_view& out_row_mask,
                                              cudf::size_type out_row_mask_offset,
                                              rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  auto const total_rows = static_cast<cudf::size_type>(in_row_mask.size());

  CUDF_EXPECTS(out_row_mask_offset + total_rows <= out_row_mask.size(),
               "Input and output row mask columns must have the same number of rows");
  CUDF_EXPECTS(out_row_mask.type().id() == type_id::BOOL8,
               "Output row mask column must be a boolean column");
  CUDF_EXPECTS(in_row_mask.type().id() == type_id::BOOL8,
               "Input row mask column must be a boolean column");

  // Update output row mask such that out_row_mask[i] = true, iff in_row_mask[i] is valid and true.
  // This is inline with the masking behavior of cudf::detail::apply_boolean_mask.
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator(total_rows),
                    out_row_mask.begin<bool>() + out_row_mask_offset,
                    [is_nullable = in_row_mask.nullable(),
                     in_row_mask = in_row_mask.begin<bool>(),
                     in_bitmask  = in_row_mask.null_mask()] __device__(auto row_idx) {
                      auto const is_valid = not is_nullable or bit_is_set(in_bitmask, row_idx);
                      auto const is_true  = in_row_mask[row_idx];
                      if (is_nullable) {
                        return is_valid and is_true;
                      } else {
                        return is_true;
                      }
                    });

  // Make sure the null mask of the output row mask column is all valid after the update. This is
  // to correctly assess if a payload column data page can be pruned. An invalid row in the row mask
  // column means the corresponding data page cannot be pruned.
  if (out_row_mask.nullable()) {
    cudf::set_null_mask(out_row_mask.null_mask(), 0, total_rows, true, stream);
    out_row_mask.set_null_count(0);
  }
}

}  // namespace cudf::io::parquet::experimental::detail
