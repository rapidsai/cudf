/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl.hpp"
#include "reader_impl_chunking.hpp"
#include "reader_impl_chunking_utils.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/gather.h>
#include <thrust/transform_scan.h>

#include <numeric>

namespace cudf::io::parquet::detail {

namespace {

// The minimum amount of memory we can safely expect to be enough to
// do a subpass decode. If the difference between the user specified limit and
// the actual memory used for compressed/temp data is > than this value, we will still use
// at least this many additional bytes.
// Example:
// - user has specified 1 GB limit
// - we have read in 900 MB of compressed data
// - that leaves us 100 MB of space for decompression batches
// - to keep the gpu busy, we really don't want to do less than 200 MB at a time so we're just going
// to use 200 MB of space
//   even if that goes past the user-specified limit.
constexpr size_t minimum_subpass_expected_size = 200 * 1024 * 1024;

// Percentage of the total available input read limit that should be reserved for compressed
// data vs uncompressed data.
constexpr float input_limit_compression_reserve = 0.3f;

}  // namespace

void reader_impl::handle_chunking(read_mode mode)
{
  // if this is our first time in here, setup the first pass.
  if (!_pass_itm_data) {
    // setup the next pass
    setup_next_pass(mode);
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
      setup_next_pass(mode);
    }
  }

  // setup the next sub pass
  setup_next_subpass(mode);
}

void reader_impl::setup_next_pass(read_mode mode)
{
  auto const num_passes = _file_itm_data.num_passes();

  // always create the pass struct, even if we end up with no work.
  // this will also cause the previous pass information to be deleted
  _pass_itm_data = std::make_unique<pass_intermediate_data>();

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

    pass.chunks = cudf::detail::hostdevice_vector<ColumnChunkDesc>(num_chunks, _stream);
    std::copy(chunk_start, chunk_end, pass.chunks.begin());

    // compute skip_rows / num_rows for this pass.
    if (num_passes == 1) {
      pass.skip_rows = _file_itm_data.global_skip_rows;
      pass.num_rows  = _file_itm_data.global_num_rows;
    } else {
      // pass_start_row and pass_end_row are computed from the selected row groups relative to the
      // global_skip_rows.
      auto const pass_start_row =
        _file_itm_data.input_pass_start_row_count[_file_itm_data._current_input_pass];
      auto const pass_end_row =
        std::min(_file_itm_data.input_pass_start_row_count[_file_itm_data._current_input_pass + 1],
                 _file_itm_data.global_num_rows);

      // pass.skip_rows is always global in the sense that it is relative to the first row of
      // the data source (global row number 0), regardless of what pass we are on. Therefore,
      // we must re-add global_skip_rows to the pass_start_row which is relative to the
      // global_skip_rows.
      pass.skip_rows = _file_itm_data.global_skip_rows + pass_start_row;
      // num_rows is how many rows we are reading this pass. Since this is a difference, adding
      // global_skip_rows to both variables is redundant.
      pass.num_rows = pass_end_row - pass_start_row;
    }

    // load page information for the chunk. this retrieves the compressed bytes for all the
    // pages, and their headers (which we can access without decompressing)
    read_compressed_data();

    // detect malformed columns.
    // - we have seen some cases in the wild where we have a row group containing N
    //   rows, but the total number of rows in the pages for column X is != N. while it
    //   is possible to load this by just capping the number of rows read, we cannot tell
    //   which rows are invalid so we may be returning bad data. in addition, this mismatch
    //   confuses the chunked reader
    detect_malformed_pages(
      pass.pages,
      pass.chunks,
      uses_custom_row_bounds(mode) ? std::nullopt : std::make_optional(pass.num_rows),
      _stream);

    // Get the decompressed size of dictionary pages to help estimate memory usage
    auto const decomp_dict_data_size = std::accumulate(
      pass.pages.begin(), pass.pages.end(), size_t{0}, [&pass](size_t acc, auto const& page) {
        if (pass.chunks[page.chunk_idx].codec != Compression::UNCOMPRESSED &&
            (page.flags & PAGEINFO_FLAGS_DICTIONARY)) {
          return acc + page.uncompressed_page_size;
        }
        return acc;
      });

    // store off how much memory we've used so far. This includes the compressed page data and the
    // decompressed dictionary data. we will subtract this from the available total memory for the
    // subpasses
    auto chunk_iter =
      thrust::make_transform_iterator(pass.chunks.d_begin(), get_chunk_compressed_size{});
    pass.base_mem_size =
      decomp_dict_data_size +
      thrust::reduce(rmm::exec_policy(_stream), chunk_iter, chunk_iter + pass.chunks.size());

    // if we are doing subpass reading, generate more accurate num_row estimates for list columns.
    // this helps us to generate more accurate subpass splits.
    if (pass.has_compressed_data && _input_pass_read_limit != 0) {
      if (not _has_page_index) {
        generate_list_column_row_counts(is_estimate_row_counts::YES);
      } else {
        generate_list_column_row_counts(is_estimate_row_counts::NO);
      }
    }

#if defined(PARQUET_CHUNK_LOGGING)
    printf("Pass: row_groups(%'lu), chunks(%'lu), pages(%'lu)\n",
           pass.row_groups.size(),
           pass.chunks.size(),
           pass.pages.size());
    printf("\tskip_rows: %'lu\n", pass.skip_rows);
    printf("\tnum_rows: %'lu\n", pass.num_rows);
    printf("\tbase mem usage: %'lu\n", pass.base_mem_size);
    auto const num_columns    = _input_columns.size();
    auto const h_page_offsets = cudf::detail::make_host_vector(pass.page_offsets, _stream);
    for (size_t c_idx = 0; c_idx < num_columns; c_idx++) {
      printf("\t\tColumn %'lu: num_pages(%'d)\n",
             c_idx,
             h_page_offsets[c_idx + 1] - h_page_offsets[c_idx]);
    }
#endif

    _stream.synchronize();
  }
}

void reader_impl::setup_next_subpass(read_mode mode)
{
  auto& pass    = *_pass_itm_data;
  pass.subpass  = std::make_unique<subpass_intermediate_data>();
  auto& subpass = *pass.subpass;

  auto const num_columns = _input_columns.size();

  // if the user has passed a very small value (under the hardcoded minimum_subpass_expected_size),
  // respect it.
  auto const min_subpass_size = std::min(_input_pass_read_limit, minimum_subpass_expected_size);

  // Check if this the first subpass in this pass
  auto const is_first_subpass = pass.processed_rows == 0;

  // what do we do if the base memory size (the compressed data) itself is approaching or larger
  // than the overall read limit? we are still going to be decompressing in subpasses, but we have
  // to assume some reasonable minimum size needed to safely decompress a single subpass. so always
  // reserve at least that much space. this can result in using up to 2x the specified user limit
  // but should only ever happen with unrealistically low numbers.
  size_t const remaining_read_limit =
    _input_pass_read_limit == 0 ? 0
    : pass.base_mem_size + min_subpass_size >= _input_pass_read_limit
      ? min_subpass_size
      : _input_pass_read_limit - pass.base_mem_size;

  // page_indices is an array of spans where each element N is the
  // indices into the pass.pages array that represents the subset of pages
  // for column N to use for the subpass.
  auto [page_indices, total_pages, total_expected_size] =
    [&]() -> std::tuple<rmm::device_uvector<page_span>, size_t, size_t> {
    if (!pass.has_compressed_data || _input_pass_read_limit == 0) {
      rmm::device_uvector<page_span> page_indices(
        num_columns, _stream, cudf::get_current_device_resource_ref());
      auto iter = thrust::make_counting_iterator(0);
      thrust::transform(rmm::exec_policy_nosync(_stream),
                        iter,
                        iter + num_columns,
                        page_indices.begin(),
                        get_page_span_by_column{pass.page_offsets});
      return {std::move(page_indices), pass.pages.size(), size_t{0}};
    }
    // otherwise we have to look forward and choose a batch of pages

    // as subpasses get decoded, the initial estimates we have for list row counts
    // get updated with accurate data, so regenerate cumulative size info and row
    // indices
    rmm::device_uvector<cumulative_page_info> c_info(pass.pages.size(), _stream);
    auto page_keys = make_page_key_iterator(pass.pages);
    auto page_size = thrust::make_transform_iterator(pass.pages.d_begin(), get_page_input_size{});
    thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(_stream),
                                  page_keys,
                                  page_keys + pass.pages.size(),
                                  page_size,
                                  c_info.begin(),
                                  cuda::std::equal_to{},
                                  cumulative_page_sum{});

    // include scratch space needed for decompression and string offset buffers.
    // for certain codecs (eg ZSTD) this an be considerable.
    if (is_first_subpass) {
      pass.decomp_scratch_sizes =
        compute_decompression_scratch_sizes(pass.chunks, pass.pages, _stream);
      pass.string_offset_sizes = compute_string_offset_sizes(pass.chunks, pass.pages, _stream);
    }
    include_scratch_size(pass.decomp_scratch_sizes, c_info, _stream);
    include_scratch_size(pass.string_offset_sizes, c_info, _stream);

    auto iter               = thrust::make_counting_iterator(0);
    auto const pass_max_row = pass.skip_rows + pass.num_rows;
    thrust::for_each(rmm::exec_policy_nosync(_stream),
                     iter,
                     iter + pass.pages.size(),
                     set_row_index{pass.chunks, pass.pages, c_info, pass_max_row});
    // print_cumulative_page_info(pass.pages, pass.chunks, c_info, _stream);

    // get the next batch of pages
    return compute_next_subpass(c_info,
                                pass.pages,
                                pass.chunks,
                                pass.page_offsets,
                                pass.processed_rows + pass.skip_rows,
                                remaining_read_limit,
                                num_columns,
                                is_first_subpass,
                                _has_page_index,
                                _stream);
  }();

  // check to see if we are processing the entire pass (enabling us to skip a bunch of work)
  subpass.single_subpass = total_pages == pass.pages.size();

  // in the single pass case, no page copying is necessary - just use what's in the pass itself
  if (subpass.single_subpass) {
    subpass.pages = pass.pages;
  }
  // copy the appropriate subset of pages from each column and store the mapping back to the source
  // (pass) pages
  else {
    subpass.page_buf       = cudf::detail::hostdevice_vector<PageInfo>(total_pages, _stream);
    subpass.page_src_index = rmm::device_uvector<size_t>(total_pages, _stream);
    auto iter              = thrust::make_counting_iterator(0);
    rmm::device_uvector<size_t> dst_offsets(num_columns + 1, _stream);
    thrust::transform_exclusive_scan(rmm::exec_policy_nosync(_stream),
                                     iter,
                                     iter + num_columns + 1,
                                     dst_offsets.begin(),
                                     get_span_size_by_index{page_indices},
                                     0,
                                     cuda::std::plus<size_t>{});
    thrust::for_each(
      rmm::exec_policy_nosync(_stream),
      iter,
      iter + total_pages,
      copy_subpass_page{
        pass.pages, subpass.page_buf, subpass.page_src_index, dst_offsets, page_indices});
    subpass.pages = subpass.page_buf;
  }

  auto const h_spans = cudf::detail::make_host_vector_async(page_indices, _stream);
  subpass.pages.device_to_host_async(_stream);

  _stream.synchronize();

  subpass.column_page_count = std::vector<size_t>(num_columns);
  std::transform(
    h_spans.begin(), h_spans.end(), subpass.column_page_count.begin(), get_span_size{});

  // Set the page mask information for the subpass
  set_subpass_page_mask();
  _subpass_page_mask.host_to_device_async(_stream);

  // decompress the data pages in this subpass; also decompress the dictionary pages in this pass,
  // if this is the first subpass in the pass
  if (pass.has_compressed_data) {
    auto [pass_data, subpass_data] =
      decompress_page_data(pass.chunks,
                           is_first_subpass ? pass.pages : host_span<PageInfo>{},
                           subpass.pages,
                           _subpass_page_mask,
                           _stream,
                           _mr);

    if (is_first_subpass) {
      pass.decomp_dict_data = std::move(pass_data);
      pass.pages.host_to_device_async(_stream);
    }

    subpass.decomp_page_data = std::move(subpass_data);
    subpass.pages.host_to_device_async(_stream);
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
  preprocess_subpass_pages(mode, _output_chunk_read_limit);

#if defined(PARQUET_CHUNK_LOGGING)
  printf("\tSubpass: skip_rows(%'lu), num_rows(%'lu), remaining read limit(%'lu)\n",
         subpass.skip_rows,
         subpass.num_rows,
         remaining_read_limit);
  printf("\t\tDecompressed size: %'lu\n", subpass.decomp_page_data.size());
  printf("\t\tTotal expected usage: %'lu\n",
         total_expected_size == 0 ? subpass.decomp_page_data.size() + pass.base_mem_size
                                  : total_expected_size + pass.base_mem_size);
  auto const h_page_indices = cudf::detail::make_host_vector(page_indices, _stream);
  for (size_t c_idx = 0; c_idx < num_columns; c_idx++) {
    printf("\t\tColumn %'lu: pages(%'lu - %'lu)\n",
           c_idx,
           h_page_indices[c_idx].start,
           h_page_indices[c_idx].end);
  }
  printf("\t\tOutput chunks:\n");
  for (size_t idx = 0; idx < subpass.output_chunk_read_info.size(); idx++) {
    printf("\t\t\t%'lu: skip_rows(%'lu) num_rows(%'lu)\n",
           idx,
           subpass.output_chunk_read_info[idx].skip_rows,
           subpass.output_chunk_read_info[idx].num_rows);
  }
#endif
}

void reader_impl::create_global_chunk_info()
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
    auto const& columns = _metadata->get_row_group(rg.index, rg.source_index).columns;
    column_mapping.resize(num_input_columns);
    std::transform(
      _input_columns.begin(), _input_columns.end(), column_mapping.begin(), [&](auto const& col) {
        // translate schema_idx into something we can use for the page indexes
        if (auto it = std::find_if(columns.begin(),
                                   columns.end(),
                                   [&](auto const& col_chunk) {
                                     return col_chunk.schema_idx ==
                                            _metadata->map_schema_index(col.schema_idx,
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
    auto const& row_group      = _metadata->get_row_group(rg.index, rg.source_index);
    auto const row_group_start = rg.start_row;
    // Adjust row_group_rows for `skip_rows` for the first row_group but cap at row_group.num_rows
    auto const adjusted_row_group_rows = skip_rows ? skip_rows - row_groups_info[0].start_row : 0;
    auto row_group_rows =
      std::min<size_t>(remaining_rows + adjusted_row_group_rows, row_group.num_rows);

    // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
    for (size_t i = 0; i < num_input_columns; ++i) {
      auto col = _input_columns[i];
      // look up metadata
      auto& col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);
      auto& schema   = _metadata->get_schema(
        _metadata->map_schema_index(col.schema_idx, rg.source_index), rg.source_index);

      auto [clock_rate, logical_type] =
        conversion_info(to_type_id(schema, _strings_to_categorical, _options.timestamp_type.id()),
                        _options.timestamp_type.id(),
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
      column_chunk_info const* const chunk_info =
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
                          _metadata->get_output_nesting_depth(col.schema_idx),
                          required_bits(schema.max_definition_level),
                          required_bits(schema.max_repetition_level),
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

void reader_impl::compute_input_passes(read_mode mode)
{
  // at this point, row_groups has already been filtered down to just the row groups we need to
  // handle optional skip_rows/num_rows parameters.
  auto const& row_groups_info = _file_itm_data.row_groups;

  // If we are reading all rows at once, read everything in a single pass. We can't use
  // `_input_pass_read_limit` as test here for `CHUNKED_READ` mode as we may need to create a pass
  // using max column size limits
  if (mode == read_mode::READ_ALL) {
    _file_itm_data.input_pass_row_group_offsets.push_back(0);
    _file_itm_data.input_pass_row_group_offsets.push_back(row_groups_info.size());
    _file_itm_data.input_pass_start_row_count.push_back(0);
    auto rg_row_count = cudf::detail::make_counting_transform_iterator(0, [&](size_t i) {
      auto const& rgi       = row_groups_info[i];
      auto const& row_group = _metadata->get_row_group(rgi.index, rgi.source_index);
      return row_group.num_rows;
    });
    _file_itm_data.input_pass_start_row_count.push_back(
      std::reduce(rg_row_count, rg_row_count + row_groups_info.size()));
    return;
  }

  // generate passes. make sure to account for the case where a single row group doesn't fit within
  // the read limit
  std::size_t const comp_read_limit =
    _input_pass_read_limit > 0
      ? static_cast<size_t>(_input_pass_read_limit * input_limit_compression_reserve)
      : std::numeric_limits<std::size_t>::max();

  // Maximum number of rows we can read in a single pass is bounded by cudf's column size limit
  auto constexpr max_rows_per_pass =
    static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());

  std::size_t cur_pass_byte_size          = 0;
  std::size_t cur_pass_num_leaf_values    = 0;
  std::size_t cur_pass_num_top_level_rows = 0;
  std::size_t cur_rg_start                = 0;
  std::size_t cur_row_count               = 0;
  _file_itm_data.input_pass_row_group_offsets.push_back(0);
  _file_itm_data.input_pass_start_row_count.push_back(0);

  // To handle global_skip_rows when computing input passes
  int64_t skip_rows = _file_itm_data.global_skip_rows;

  for (size_t cur_rg_index = 0; cur_rg_index < row_groups_info.size(); cur_rg_index++) {
    auto const& rgi       = row_groups_info[cur_rg_index];
    auto const& row_group = _metadata->get_row_group(rgi.index, rgi.source_index);

    // total compressed size and total size (compressed + uncompressed) for
    auto const [compressed_rg_size, _ /*compressed + uncompressed*/] =
      get_row_group_size(row_group);

    // We must use the effective size of the first row group we are reading to accurately calculate
    // the first non-zero `input_pass_start_row_count` unless we are reading only one row group
    auto const row_group_rows = (skip_rows and row_groups_info.size() > 1)
                                  ? (rgi.start_row + row_group.num_rows - skip_rows)
                                  : row_group.num_rows;

    // Get the number of leaf-level number of values in this row group. Note that this value may
    // not represent the number of leaf-level rows as it does not account for nulls
    auto const row_group_leaf_values =
      std::max_element(row_group.columns.cbegin(),
                       row_group.columns.cend(),
                       [](auto const& a, auto const& b) {
                         return a.meta_data.num_values < b.meta_data.num_values;
                       })
        ->meta_data.num_values;

    //  Set skip_rows = 0 as it is no longer needed for subsequent row_groups
    skip_rows = 0;

    // Check if we need to create a pass boundary here?
    // Note: Here we may end up with an invalid pass (number of rows exceeding the cudf column size
    // limit) in certain edge case conditions such as:
    // 1. Number of leaf-level values plus nulls (computed by dremel decoding) exceeds the cudf
    // column size limit
    // 2. For nested lists (list<list<list<...>>>), one or more nested list(s) may have number of
    // rows (computed by dremel decoding) exceeding the cudf column size limit
    if ((cur_pass_byte_size + compressed_rg_size >= comp_read_limit) or
        (cur_pass_num_leaf_values + row_group_leaf_values >= max_rows_per_pass) or
        (cur_pass_num_top_level_rows + row_group_rows >= max_rows_per_pass)) {
      // A single row group (the current one) is larger than the read limit:
      // We always need to include at least one row group, so end the pass at the end of the current
      // row group
      if (cur_rg_start == cur_rg_index) {
        CUDF_EXPECTS(std::cmp_less_equal(row_group.num_rows, max_rows_per_pass),
                     "Number of rows in each row group must be smaller than the column size limit");
        _file_itm_data.input_pass_row_group_offsets.push_back(cur_rg_index + 1);
        _file_itm_data.input_pass_start_row_count.push_back(cur_row_count + row_group_rows);
        cur_rg_start                = cur_rg_index + 1;
        cur_pass_byte_size          = 0;
        cur_pass_num_leaf_values    = 0;
        cur_pass_num_top_level_rows = 0;
      }
      // End the pass at the end of the previous row group
      else {
        _file_itm_data.input_pass_row_group_offsets.push_back(cur_rg_index);
        _file_itm_data.input_pass_start_row_count.push_back(cur_row_count);
        cur_rg_start                = cur_rg_index;
        cur_pass_byte_size          = compressed_rg_size;
        cur_pass_num_leaf_values    = row_group_leaf_values;
        cur_pass_num_top_level_rows = row_group_rows;
      }
    } else {
      cur_pass_byte_size += compressed_rg_size;
      cur_pass_num_leaf_values += row_group_leaf_values;
      cur_pass_num_top_level_rows += row_group_rows;
    }
    cur_row_count += row_group_rows;
  }

  // add the last pass if necessary
  if (_file_itm_data.input_pass_row_group_offsets.back() != row_groups_info.size()) {
    _file_itm_data.input_pass_row_group_offsets.push_back(row_groups_info.size());
    _file_itm_data.input_pass_start_row_count.push_back(cur_row_count);
  }
}

void reader_impl::compute_output_chunks_for_subpass()
{
  CUDF_FUNC_RANGE();

  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  // simple case : no chunk size, no splits
  if (_output_chunk_read_limit <= 0) {
    subpass.output_chunk_read_info.push_back({subpass.skip_rows, subpass.num_rows});
    return;
  }

  // generate row_indices and cumulative output sizes for all pages
  rmm::device_uvector<cumulative_page_info> c_info(subpass.pages.size(), _stream);
  auto page_input =
    thrust::make_transform_iterator(subpass.pages.device_begin(), get_page_output_size{});
  auto page_keys = make_page_key_iterator(subpass.pages);
  thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(_stream),
                                page_keys,
                                page_keys + subpass.pages.size(),
                                page_input,
                                c_info.begin(),
                                cuda::std::equal_to{},
                                cumulative_page_sum{});
  auto iter = thrust::make_counting_iterator(0);
  // cap the max row in all pages by the max row we expect in the subpass. input chunking
  // can cause "dangling" row counts where for example, only 1 column has a page whose
  // maximum row is beyond our expected subpass max row, which will cause an out of
  // bounds index in compute_page_splits_by_row.
  auto const subpass_max_row = subpass.skip_rows + subpass.num_rows;
  thrust::for_each(rmm::exec_policy_nosync(_stream),
                   iter,
                   iter + subpass.pages.size(),
                   set_row_index{pass.chunks, subpass.pages, c_info, subpass_max_row});
  // print_cumulative_page_info(subpass.pages, pass.chunks, c_info, _stream);

  // compute the splits
  subpass.output_chunk_read_info = compute_page_splits_by_row(
    c_info, subpass.pages, subpass.skip_rows, subpass.num_rows, _output_chunk_read_limit, _stream);
}

void reader_impl::set_subpass_page_mask()
{
  auto const& pass    = _pass_itm_data;
  auto const& subpass = pass->subpass;

  // Create a hostdevice vector to store the subpass page mask
  _subpass_page_mask = cudf::detail::hostdevice_vector<bool>(subpass->pages.size(), _stream);

  // Fill with all true if no pass level page mask is available
  if (_pass_page_mask.empty()) {
    std::fill(_subpass_page_mask.begin(), _subpass_page_mask.end(), true);
    return;
  }

  // If this is the only subpass, move the pass level page mask data as is
  if (subpass->single_subpass) {
    std::move(_pass_page_mask.begin(), _pass_page_mask.end(), _subpass_page_mask.begin());
    return;
  }

  // Use the pass page index mask to gather the subpass page mask from the pass level page mask
  auto const host_page_src_index = cudf::detail::make_host_vector(subpass->page_src_index, _stream);
  thrust::gather(thrust::seq,
                 host_page_src_index.begin(),
                 host_page_src_index.end(),
                 _pass_page_mask.begin(),
                 _subpass_page_mask.begin());
}

}  // namespace cudf::io::parquet::detail
