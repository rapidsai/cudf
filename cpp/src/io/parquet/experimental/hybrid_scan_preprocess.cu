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

#include "hybrid_scan_helpers.hpp"
#include "hybrid_scan_impl.hpp"
#include "io/parquet/parquet_gpu.hpp"
#include "io/parquet/reader_impl_chunking_utils.cuh"
#include "io/parquet/reader_impl_preprocess_utils.cuh"

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
using parquet::detail::input_col_info;
using parquet::detail::level_type;
using parquet::detail::PageInfo;
using parquet::detail::PageNestingDecodeInfo;
using parquet::detail::PageNestingInfo;

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
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  parquet_reader_options const& options)
{
  // Hybrid scan reader does not support skip rows
  _file_itm_data.global_skip_rows = 0;

  std::tie(_file_itm_data.global_num_rows, _file_itm_data.row_groups) =
    _extended_metadata->select_row_groups(row_group_indices);

  // check for page indexes
  _has_page_index = std::all_of(_file_itm_data.row_groups.cbegin(),
                                _file_itm_data.row_groups.cend(),
                                [](auto const& row_group) { return row_group.has_page_index(); });

  if (_file_itm_data.global_num_rows > 0 && not _file_itm_data.row_groups.empty() &&
      not _input_columns.empty()) {
    // fills in chunk information without physically loading or decompressing
    // the associated data
    create_global_chunk_info(options);

    // compute schedule of input reads.
    compute_input_passes();
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
  std::vector<rmm::device_buffer> column_chunk_buffers)
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

void hybrid_scan_reader_impl::allocate_columns(size_t skip_rows, size_t num_rows)
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  // Should not reach here if there is no page data.
  CUDF_EXPECTS(subpass.pages.size() > 0, "There are no pages present in the subpass");

  // computes:
  // PageNestingInfo::batch_size for each level of nesting, for each page, taking row bounds into
  // account. PageInfo::skipped_values, which tells us where to start decoding in the input to
  // respect the user bounds. It is only necessary to do this second pass if uses_custom_row_bounds
  // is set (if the user has specified artificial bounds).
  if (uses_custom_row_bounds(read_mode::READ_ALL)) {
    parquet::detail::compute_page_sizes(subpass.pages,
                                        pass.chunks,
                                        skip_rows,
                                        num_rows,
                                        false,  // num_rows is already computed
                                        false,  // no need to compute string sizes
                                        pass.level_type_size,
                                        _stream);
  }

  // iterate over all input columns and allocate any associated output
  // buffers if they are not part of a list hierarchy. mark down
  // if we have any list columns that need further processing.
  bool has_lists = false;
  // Casting to std::byte since data buffer pointer is void *
  std::vector<cudf::device_span<cuda::std::byte>> memset_bufs;
  // Validity Buffer is a uint32_t pointer
  std::vector<cudf::device_span<cudf::bitmask_type>> nullmask_bufs;

  for (auto const& input_col : _input_columns) {
    size_t const max_depth = input_col.nesting_depth();

    auto* cols = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // if this has a list parent, we have to get column sizes from the
      // data computed during ComputePageSizes
      if (out_buf.user_data &
          cudf::io::parquet::detail::PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) {
        has_lists = true;
      }
      // if we haven't already processed this column because it is part of a struct hierarchy
      else if (out_buf.size == 0) {
        // add 1 for the offset if this is a list column
        // we're going to start null mask as all valid and then turn bits off if necessary
        out_buf.create_with_mask(
          out_buf.type.id() == type_id::LIST && l_idx < max_depth ? num_rows + 1 : num_rows,
          cudf::mask_state::UNINITIALIZED,
          false,
          _stream,
          _mr);
        memset_bufs.push_back(cudf::device_span<cuda::std::byte>(
          static_cast<cuda::std::byte*>(out_buf.data()), out_buf.data_size()));
        nullmask_bufs.push_back(cudf::device_span<cudf::bitmask_type>(
          out_buf.null_mask(),
          cudf::util::round_up_safe(out_buf.null_mask_size(), sizeof(cudf::bitmask_type)) /
            sizeof(cudf::bitmask_type)));
      }
    }
  }
  // compute output column sizes by examining the pages of the -input- columns
  if (has_lists) {
    auto h_cols_info =
      cudf::detail::make_empty_host_vector<input_col_info>(_input_columns.size(), _stream);
    std::transform(_input_columns.cbegin(),
                   _input_columns.cend(),
                   std::back_inserter(h_cols_info),
                   [](auto& col) -> input_col_info {
                     return {col.schema_idx, static_cast<size_type>(col.nesting_depth())};
                   });

    auto const max_depth =
      (*std::max_element(h_cols_info.cbegin(),
                         h_cols_info.cend(),
                         [](auto& l, auto& r) { return l.nesting_depth < r.nesting_depth; }))
        .nesting_depth;

    auto const d_cols_info = cudf::detail::make_device_uvector_async(
      h_cols_info, _stream, cudf::get_current_device_resource_ref());

    // Vector to store page sizes for each column at each depth
    cudf::detail::hostdevice_vector<size_t> sizes{_input_columns.size() * max_depth, _stream};

    // Total number of keys to process
    auto const num_keys = _input_columns.size() * max_depth * subpass.pages.size();

    // Maximum 1 billion keys processed per iteration
    auto constexpr max_keys_per_iter =
      static_cast<size_t>(std::numeric_limits<size_type>::max() / 2);

    // Number of keys for per each column
    auto const num_keys_per_col = max_depth * subpass.pages.size();

    // The largest multiple of `num_keys_per_col` that is <= `num_keys`
    auto const num_keys_per_iter =
      num_keys <= max_keys_per_iter
        ? num_keys
        : num_keys_per_col * std::max<size_t>(1, max_keys_per_iter / num_keys_per_col);

    // Size iterator. Indexes pages by sorted order
    rmm::device_uvector<size_type> size_input{num_keys_per_iter, _stream};

    // To keep track of the starting key of an iteration
    size_t key_start = 0;
    // Loop until all keys are processed
    while (key_start < num_keys) {
      // Number of keys processed in this iteration
      auto const num_keys_this_iter = std::min<size_t>(num_keys_per_iter, num_keys - key_start);
      thrust::transform(
        rmm::exec_policy_nosync(_stream),
        thrust::make_counting_iterator<size_t>(key_start),
        thrust::make_counting_iterator<size_t>(key_start + num_keys_this_iter),
        size_input.begin(),
        parquet::detail::get_page_nesting_size{
          d_cols_info.data(), max_depth, subpass.pages.size(), subpass.pages.device_begin()});

      // Manually create a size_t `key_start` compatible counting_transform_iterator.
      auto const reduction_keys =
        thrust::make_transform_iterator(thrust::make_counting_iterator<std::size_t>(key_start),
                                        parquet::detail::get_reduction_key{subpass.pages.size()});

      // Find the size of each column
      thrust::reduce_by_key(rmm::exec_policy_nosync(_stream),
                            reduction_keys,
                            reduction_keys + num_keys_this_iter,
                            size_input.cbegin(),
                            thrust::make_discard_iterator(),
                            sizes.d_begin() + (key_start / subpass.pages.size()));

      // For nested hierarchies, compute per-page start offset
      thrust::exclusive_scan_by_key(
        rmm::exec_policy_nosync(_stream),
        reduction_keys,
        reduction_keys + num_keys_this_iter,
        size_input.cbegin(),
        parquet::detail::start_offset_output_iterator{subpass.pages.device_begin(),
                                                      key_start,
                                                      d_cols_info.data(),
                                                      max_depth,
                                                      subpass.pages.size()});
      // Increment the key_start
      key_start += num_keys_this_iter;
    }

    sizes.device_to_host(_stream);
    for (size_type idx = 0; idx < static_cast<size_type>(_input_columns.size()); idx++) {
      auto const& input_col = _input_columns[idx];
      auto* cols            = &_output_buffers;
      for (size_type l_idx = 0; l_idx < static_cast<size_type>(input_col.nesting_depth());
           l_idx++) {
        auto& out_buf = (*cols)[input_col.nesting[l_idx]];
        cols          = &out_buf.children;
        // if this buffer is part of a list hierarchy, we need to determine it's
        // final size and allocate it here.
        //
        // for struct columns, higher levels of the output columns are shared between input
        // columns. so don't compute any given level more than once.
        if ((out_buf.user_data &
             cudf::io::parquet::detail::PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) &&
            out_buf.size == 0) {
          auto size = sizes[(idx * max_depth) + l_idx];

          // if this is a list column add 1 for non-leaf levels for the terminating offset
          if (out_buf.type.id() == type_id::LIST && l_idx < max_depth) { size++; }

          // allocate
          // we're going to start null mask as all valid and then turn bits off if necessary
          out_buf.create_with_mask(size, cudf::mask_state::UNINITIALIZED, false, _stream, _mr);
          memset_bufs.push_back(cudf::device_span<cuda::std::byte>(
            static_cast<cuda::std::byte*>(out_buf.data()), out_buf.data_size()));
          nullmask_bufs.push_back(cudf::device_span<cudf::bitmask_type>(
            out_buf.null_mask(),
            cudf::util::round_up_safe(out_buf.null_mask_size(), sizeof(cudf::bitmask_type)) /
              sizeof(cudf::bitmask_type)));
        }
      }
    }
  }

  cudf::detail::batched_memset<cuda::std::byte>(
    memset_bufs, static_cast<cuda::std::byte>(0), _stream);
  // Need to set null mask bufs to all high bits
  cudf::detail::batched_memset<cudf::bitmask_type>(
    nullmask_bufs, std::numeric_limits<cudf::bitmask_type>::max(), _stream);
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
  auto const row_groups_info =
    std::get<1>(_extended_metadata->select_row_groups(row_group_indices));

  CUDF_EXPECTS(row_groups_info.size() * _input_columns.size() == dictionary_page_data.size(),
               "Dictionary page data size must match the number of row groups times the number of "
               "input columns");

  // Number of input columns
  auto const num_input_columns = _input_columns.size();
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

      auto const [clock_rate, _] = parquet::detail::conversion_info(
        parquet::detail::to_type_id(schema,
                                    options.is_enabled_convert_strings_to_categories(),
                                    options.get_timestamp_type().id()),
        options.get_timestamp_type().id(),
        schema.type,
        schema.logical_type);

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

}  // namespace cudf::io::parquet::experimental::detail
