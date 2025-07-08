
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
#include "io/parquet/reader_impl_chunking.hpp"
#include "io/parquet/reader_impl_chunking_utils.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/transform_scan.h>

#include <numeric>

namespace cudf::io::parquet::experimental::detail {

using parquet::detail::ColumnChunkDesc;
using parquet::detail::CompactProtocolReader;
using parquet::detail::level_type;
using parquet::detail::page_span;
using parquet::detail::PageInfo;

void hybrid_scan_reader_impl::create_global_chunk_info(parquet_reader_options const& options)
{
  auto const num_rows         = _file_itm_data.global_num_rows;
  auto const& row_groups_info = _file_itm_data.row_groups;
  auto& chunks                = _file_itm_data.chunks;

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_chunks        = row_groups_info.size() * num_input_columns;

  // Mapping of input column to page index column
  std::vector<size_type> column_mapping;

  if (_has_page_index and not row_groups_info.empty()) {
    // use first row group to define mappings (assumes same schema for each file)
    auto const& rg      = row_groups_info[0];
    auto const& columns = _extended_metadata->get_row_group(rg.index, rg.source_index).columns;
    column_mapping.resize(num_input_columns);
    std::transform(
      _input_columns.begin(), _input_columns.end(), column_mapping.begin(), [&](auto const& col) {
        // translate schema_idx into something we can use for the page indexes
        if (auto it = std::find_if(columns.begin(),
                                   columns.end(),
                                   [&](auto const& col_chunk) {
                                     return col_chunk.schema_idx ==
                                            _extended_metadata->map_schema_index(col.schema_idx,
                                                                                 rg.source_index);
                                   });
            it != columns.end()) {
          return std::distance(columns.begin(), it);
        }
        CUDF_FAIL("cannot find column mapping");
      });
  }

  // Initialize column chunk information
  auto remaining_rows = num_rows;
  auto skip_rows      = _file_itm_data.global_skip_rows;
  for (auto const& rg : row_groups_info) {
    auto const& row_group      = _extended_metadata->get_row_group(rg.index, rg.source_index);
    auto const row_group_start = rg.start_row;
    auto const row_group_rows  = std::min<int>(remaining_rows, row_group.num_rows);

    // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
    for (size_t i = 0; i < num_input_columns; ++i) {
      auto col = _input_columns[i];
      // look up metadata
      auto& col_meta =
        _extended_metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);
      auto& schema = _extended_metadata->get_schema(
        _extended_metadata->map_schema_index(col.schema_idx, rg.source_index), rg.source_index);

      auto [clock_rate, logical_type] = parquet::detail::conversion_info(
        cudf::io::parquet::detail::to_type_id(schema,
                                              options.is_enabled_convert_strings_to_categories(),
                                              options.get_timestamp_type().id()),
        options.get_timestamp_type().id(),
        schema.type,
        schema.logical_type);

      // for lists, estimate the number of bytes per row. this is used by the subpass reader to
      // determine where to split the decompression boundaries
      float const list_bytes_per_row_est =
        schema.max_repetition_level > 0 && row_group.num_rows > 0
          ? static_cast<float>(col_meta.total_uncompressed_size) /
              static_cast<float>(row_group.num_rows)
          : 0.0f;

      // grab the column_chunk_info for each chunk (if it exists)
      cudf::io::parquet::detail::column_chunk_info const* const chunk_info =
        _has_page_index ? &rg.column_chunks.value()[column_mapping[i]] : nullptr;

      chunks.emplace_back(col_meta.total_compressed_size,
                          nullptr,
                          col_meta.num_values,
                          schema.type,
                          schema.type_length,
                          row_group_start,
                          row_group_rows,
                          schema.max_definition_level,
                          schema.max_repetition_level,
                          _extended_metadata->get_output_nesting_depth(col.schema_idx),
                          parquet::detail::required_bits(schema.max_definition_level),
                          parquet::detail::required_bits(schema.max_repetition_level),
                          col_meta.codec,
                          logical_type,
                          clock_rate,
                          i,
                          col.schema_idx,
                          chunk_info,
                          list_bytes_per_row_est,
                          schema.type == Type::BYTE_ARRAY and _strings_to_categorical,
                          rg.source_index);
    }
    // Adjust for skip_rows when updating the remaining rows after the first group
    remaining_rows -=
      (skip_rows) ? std::min<int>(rg.start_row + row_group.num_rows - skip_rows, remaining_rows)
                  : row_group_rows;
    // Set skip_rows = 0 as it is no longer needed for subsequent row_groups
    skip_rows = 0;
  }
}

void hybrid_scan_reader_impl::handle_chunking(
  std::vector<rmm::device_buffer> column_chunk_buffers,
  cudf::host_span<thrust::host_vector<bool> const> data_page_mask,
  parquet_reader_options const& options)
{
  // if this is our first time in here, setup the first pass.
  if (!_pass_itm_data) {
    // setup the next pass
    setup_next_pass(std::move(column_chunk_buffers), options);
  }

  auto& pass = *_pass_itm_data;

  // if we already have a subpass in flight.
  if (pass.subpass != nullptr) {
    // if it still has more chunks in flight, there's nothing more to do
    if (pass.subpass->current_output_chunk < pass.subpass->output_chunk_read_info.size()) {
      return;
    }

    // increment rows processed
    pass.processed_rows += pass.subpass->num_rows;

    // release the old subpass (will free memory)
    pass.subpass.reset();

    // otherwise we are done with the pass entirely
    if (pass.processed_rows == pass.num_rows) {
      // release the old pass
      _pass_itm_data.reset();

      _file_itm_data._current_input_pass++;
      // no more passes. we are absolutely done with this file.
      if (_file_itm_data._current_input_pass == _file_itm_data.num_passes()) { return; }

      // setup the next pass
      setup_next_pass(std::move(column_chunk_buffers), options);
    }
  }

  // Must be called before `setup_next_subpass()` to select pages to decompress
  set_page_mask(data_page_mask);

  // setup the next sub pass
  setup_next_subpass(options);
}

void hybrid_scan_reader_impl::setup_next_pass(std::vector<rmm::device_buffer> column_chunk_buffers,
                                              parquet_reader_options const& options)
{
  auto const num_passes = _file_itm_data.num_passes();
  CUDF_EXPECTS(num_passes == 1, "");

  // always create the pass struct, even if we end up with no work.
  // this will also cause the previous pass information to be deleted
  _pass_itm_data = std::make_unique<cudf::io::parquet::detail::pass_intermediate_data>();

  if (_file_itm_data.global_num_rows > 0 && not _file_itm_data.row_groups.empty() &&
      not _input_columns.empty() && _file_itm_data._current_input_pass < num_passes) {
    auto& pass = *_pass_itm_data;

    // setup row groups to be loaded for this pass
    auto const row_group_start =
      _file_itm_data.input_pass_row_group_offsets[_file_itm_data._current_input_pass];
    auto const row_group_end =
      _file_itm_data.input_pass_row_group_offsets[_file_itm_data._current_input_pass + 1];
    auto const num_row_groups = row_group_end - row_group_start;
    pass.row_groups.resize(num_row_groups);
    std::copy(_file_itm_data.row_groups.begin() + row_group_start,
              _file_itm_data.row_groups.begin() + row_group_end,
              pass.row_groups.begin());

    CUDF_EXPECTS(_file_itm_data._current_input_pass < num_passes,
                 "Encountered an invalid read pass index");

    auto const chunks_per_rowgroup = _input_columns.size();
    auto const num_chunks          = chunks_per_rowgroup * num_row_groups;

    auto chunk_start = _file_itm_data.chunks.begin() + (row_group_start * chunks_per_rowgroup);
    auto chunk_end   = _file_itm_data.chunks.begin() + (row_group_end * chunks_per_rowgroup);

    pass.chunks = cudf::detail::hostdevice_vector<cudf::io::parquet::detail::ColumnChunkDesc>(
      num_chunks, _stream);
    std::copy(chunk_start, chunk_end, pass.chunks.begin());

    // compute skip_rows / num_rows for this pass.
    pass.skip_rows = _file_itm_data.global_skip_rows;
    pass.num_rows  = _file_itm_data.global_num_rows;

    // Setup page information for the chunk (which we can access without decompressing)
    setup_compressed_data(std::move(column_chunk_buffers));

    // detect malformed columns.
    // - we have seen some cases in the wild where we have a row group containing N
    //   rows, but the total number of rows in the pages for column X is != N. while it
    //   is possible to load this by just capping the number of rows read, we cannot tell
    //   which rows are invalid so we may be returning bad data. in addition, this mismatch
    //   confuses the chunked reader
    detect_malformed_pages(pass.pages,
                           pass.chunks,
                           (options.get_num_rows().has_value() or options.get_skip_rows() > 0)
                             ? std::nullopt
                             : std::make_optional(pass.num_rows),
                           _stream);

    // Get the decompressed size of dictionary pages to help estimate memory usage
    auto const decomp_dict_data_size = std::accumulate(
      pass.pages.begin(), pass.pages.end(), size_t{0}, [&pass](size_t acc, auto const& page) {
        if (pass.chunks[page.chunk_idx].codec != Compression::UNCOMPRESSED &&
            (page.flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY)) {
          return acc + page.uncompressed_page_size;
        }
        return acc;
      });

    // store off how much memory we've used so far. This includes the compressed page data and the
    // decompressed dictionary data. we will subtract this from the available total memory for the
    // subpasses
    auto chunk_iter = thrust::make_transform_iterator(pass.chunks.d_begin(),
                                                      parquet::detail::get_chunk_compressed_size{});
    pass.base_mem_size =
      decomp_dict_data_size +
      thrust::reduce(rmm::exec_policy(_stream), chunk_iter, chunk_iter + pass.chunks.size());

    _stream.synchronize();
  }
}

void hybrid_scan_reader_impl::setup_next_subpass(parquet_reader_options const& options)
{
  auto& pass    = *_pass_itm_data;
  pass.subpass  = std::make_unique<cudf::io::parquet::detail::subpass_intermediate_data>();
  auto& subpass = *pass.subpass;

  auto const num_columns = _input_columns.size();

  // page_indices is an array of spans where each element N is the
  // indices into the pass.pages array that represents the subset of pages
  // for column N to use for the subpass.
  auto [page_indices, total_pages, total_expected_size] =
    [&]() -> std::tuple<rmm::device_uvector<page_span>, size_t, size_t> {
    rmm::device_uvector<page_span> page_indices(
      num_columns, _stream, cudf::get_current_device_resource_ref());
    auto iter = thrust::make_counting_iterator(0);
    thrust::transform(rmm::exec_policy_nosync(_stream),
                      iter,
                      iter + num_columns,
                      page_indices.begin(),
                      parquet::detail::get_page_span_by_column{pass.page_offsets});
    return {std::move(page_indices), pass.pages.size(), size_t{0}};
  }();

  // check to see if we are processing the entire pass (enabling us to skip a bunch of work)
  subpass.single_subpass = total_pages == pass.pages.size();

  CUDF_EXPECTS(subpass.single_subpass, "Hybrid scan reader must read in one subpass");

  // in the single pass case, no page copying is necessary - just use what's in the pass itself
  subpass.pages = pass.pages;

  auto const h_spans = cudf::detail::make_host_vector_async(page_indices, _stream);
  subpass.pages.device_to_host_async(_stream);

  _stream.synchronize();

  subpass.column_page_count = std::vector<size_t>(num_columns);
  std::transform(h_spans.begin(),
                 h_spans.end(),
                 subpass.column_page_count.begin(),
                 parquet::detail::get_span_size{});

  auto const is_first_subpass = pass.processed_rows == 0;

  // decompress the data pages in this subpass; also decompress the dictionary pages in this pass,
  // if this is the first subpass in the pass
  if (pass.has_compressed_data) {
    auto [pass_data, subpass_data] =
      parquet::detail::decompress_page_data(pass.chunks,
                                            is_first_subpass ? pass.pages : host_span<PageInfo>{},
                                            subpass.pages,
                                            _page_mask,
                                            _stream,
                                            _mr);

    if (is_first_subpass) {
      pass.decomp_dict_data = std::move(pass_data);
      pass.pages.host_to_device(_stream);
    }

    subpass.decomp_page_data = std::move(subpass_data);
    subpass.pages.host_to_device(_stream);
  }

  // since there is only ever 1 dictionary per chunk (the first page), do it at the
  // pass level.
  if (is_first_subpass) { build_string_dict_indices(); }

  // buffers needed by the decode kernels
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
  preprocess_subpass_pages(read_mode::READ_ALL, 0);
}

void hybrid_scan_reader_impl::update_row_mask(cudf::column_view in_row_mask,
                                              cudf::mutable_column_view out_row_mask,
                                              rmm::cuda_stream_view stream)
{
  auto const total_rows = static_cast<cudf::size_type>(in_row_mask.size());

  CUDF_EXPECTS(total_rows == out_row_mask.size(),
               "Input and output row mask columns must have the same number of rows");
  CUDF_EXPECTS(out_row_mask.type().id() == type_id::BOOL8,
               "Output row mask column must be a boolean column");

  // Update output row mask such that out_row_mask[i] = true, iff in_row_mask[i] is valid and true.
  // This is inline with the masking behavior of cudf::detail::apply_boolean_mask.
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator(total_rows),
                    out_row_mask.begin<bool>(),
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

void hybrid_scan_reader_impl::sanitize_row_mask(
  cudf::host_span<thrust::host_vector<bool> const> data_page_mask,
  cudf::mutable_column_view row_mask)
{
  if (data_page_mask.empty()) {
    thrust::fill(
      rmm::exec_policy_nosync(_stream), row_mask.begin<bool>(), row_mask.end<bool>(), true);
  }
}

}  // namespace cudf::io::parquet::experimental::detail