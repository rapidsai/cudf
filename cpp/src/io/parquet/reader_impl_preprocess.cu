/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "error.hpp"
#include "io/comp/common.hpp"
#include "reader_impl.hpp"
#include "reader_impl_preprocess_utils.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/batched_memset.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <limits>
#include <numeric>

namespace cudf::io::parquet::detail {

namespace {
// Tests the passed in logical type for a FIXED_LENGTH_BYTE_ARRAY column to see if it should
// be treated as a string. Currently the only logical type that has special handling is DECIMAL.
// Other valid types in the future would be UUID (still treated as string) and FLOAT16 (which
// for now would also be treated as a string).
inline bool is_treat_fixed_length_as_string(std::optional<LogicalType> const& logical_type)
{
  if (!logical_type.has_value()) { return true; }
  return logical_type->type != LogicalType::DECIMAL;
}

struct set_str_bytes_all {
  __device__ void operator()(PageInfo& p) { p.str_bytes_all = p.str_bytes; }
};

}  // namespace

void reader_impl::build_string_dict_indices()
{
  CUDF_FUNC_RANGE();

  auto& pass = *_pass_itm_data;

  // compute number of indices per chunk and a summed total
  rmm::device_uvector<size_t> str_dict_index_count(pass.chunks.size() + 1, _stream);
  thrust::fill(
    rmm::exec_policy_nosync(_stream), str_dict_index_count.begin(), str_dict_index_count.end(), 0);
  thrust::for_each(rmm::exec_policy_nosync(_stream),
                   pass.pages.d_begin(),
                   pass.pages.d_end(),
                   set_str_dict_index_count{str_dict_index_count, pass.chunks});

  size_t const total_str_dict_indexes = thrust::reduce(
    rmm::exec_policy(_stream), str_dict_index_count.begin(), str_dict_index_count.end());
  if (total_str_dict_indexes == 0) { return; }

  // convert to offsets
  rmm::device_uvector<size_t>& str_dict_index_offsets = str_dict_index_count;
  thrust::exclusive_scan(rmm::exec_policy_nosync(_stream),
                         str_dict_index_offsets.begin(),
                         str_dict_index_offsets.end(),
                         str_dict_index_offsets.begin(),
                         0);

  // allocate and distribute pointers
  pass.str_dict_index = cudf::detail::make_zeroed_device_uvector_async<string_index_pair>(
    total_str_dict_indexes, _stream, cudf::get_current_device_resource_ref());

  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(
    rmm::exec_policy_nosync(_stream),
    iter,
    iter + pass.chunks.size(),
    set_str_dict_index_ptr{pass.str_dict_index.data(), str_dict_index_offsets, pass.chunks});

  // compute the indices
  build_string_dictionary_index(pass.chunks.device_ptr(), pass.chunks.size(), _stream);
  pass.chunks.device_to_host(_stream);
}

void reader_impl::allocate_nesting_info()
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  auto const num_columns         = _input_columns.size();
  auto& pages                    = subpass.pages;
  auto& page_nesting_info        = subpass.page_nesting_info;
  auto& page_nesting_decode_info = subpass.page_nesting_decode_info;

  // generate the number of nesting info structs needed per-page, by column
  std::vector<int> per_page_nesting_info_size(num_columns);
  auto iter = thrust::make_counting_iterator(size_type{0});
  std::transform(iter, iter + num_columns, per_page_nesting_info_size.begin(), [&](size_type i) {
    // Schema index of the current input column
    auto const schema_idx = _input_columns[i].schema_idx;
    // Get the max_definition_level of this column across all sources.
    auto max_definition_level = _metadata->get_schema(schema_idx).max_definition_level + 1;
    std::for_each(thrust::make_counting_iterator(static_cast<size_t>(1)),
                  thrust::make_counting_iterator(_num_sources),
                  [&](auto const src_file_idx) {
                    auto const& schema = _metadata->get_schema(
                      _metadata->map_schema_index(schema_idx, src_file_idx), src_file_idx);
                    max_definition_level =
                      std::max(max_definition_level, schema.max_definition_level + 1);
                  });

    return std::max(max_definition_level, _metadata->get_output_nesting_depth(schema_idx));
  });

  // compute total # of page_nesting infos needed and allocate space. doing this in one
  // buffer to keep it to a single gpu allocation
  auto counting_iter = thrust::make_counting_iterator(size_t{0});
  size_t const total_page_nesting_infos =
    std::accumulate(counting_iter, counting_iter + num_columns, 0, [&](int total, size_t index) {
      return total + (per_page_nesting_info_size[index] * subpass.column_page_count[index]);
    });

  page_nesting_info =
    cudf::detail::hostdevice_vector<PageNestingInfo>{total_page_nesting_infos, _stream};
  page_nesting_decode_info =
    cudf::detail::hostdevice_vector<PageNestingDecodeInfo>{total_page_nesting_infos, _stream};

  // update pointers in the PageInfos
  int target_page_index = 0;
  int src_info_index    = 0;
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    auto const src_col_schema = _input_columns[idx].schema_idx;

    for (size_t p_idx = 0; p_idx < subpass.column_page_count[idx]; p_idx++) {
      pages[target_page_index + p_idx].nesting = page_nesting_info.device_ptr() + src_info_index;
      pages[target_page_index + p_idx].nesting_decode =
        page_nesting_decode_info.device_ptr() + src_info_index;

      pages[target_page_index + p_idx].nesting_info_size = per_page_nesting_info_size[idx];
      // Set the number of output nesting levels from the zeroth source as nesting must be
      // identical across sources.
      pages[target_page_index + p_idx].num_output_nesting_levels =
        _metadata->get_output_nesting_depth(src_col_schema);

      src_info_index += per_page_nesting_info_size[idx];
    }
    target_page_index += subpass.column_page_count[idx];
  }

  // Reset the target_page_index
  target_page_index = 0;

  // fill in
  int nesting_info_index = 0;
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    auto const src_col_schema = _input_columns[idx].schema_idx;

    // real depth of the output cudf column hierarchy (1 == no nesting, 2 == 1 level, etc)
    // nesting depth must be same across sources so getting it from the zeroth source is ok
    int const max_output_depth = _metadata->get_output_nesting_depth(src_col_schema);

    // Map to store depths if this column has lists
    std::map<std::pair<int, int>, std::pair<std::vector<int>, std::vector<int>>> depth_remapping;
    // if this column has lists, generate depth remapping
    std::for_each(
      thrust::make_counting_iterator(static_cast<size_t>(0)),
      thrust::make_counting_iterator(_num_sources),
      [&](auto const src_file_idx) {
        auto const mapped_schema_idx = _metadata->map_schema_index(src_col_schema, src_file_idx);
        if (_metadata->get_schema(mapped_schema_idx, src_file_idx).max_repetition_level > 0) {
          generate_depth_remappings(
            depth_remapping, src_col_schema, mapped_schema_idx, src_file_idx, *_metadata);
        }
      });

    // fill in host-side nesting info
    int schema_idx = src_col_schema;
    // This is okay as we only use this to check stubness of cur_schema and
    // to get its parent's indices, both of which are one to one mapped.
    auto cur_schema = _metadata->get_schema(schema_idx);
    int cur_depth   = max_output_depth - 1;
    while (schema_idx > 0) {
      // stub columns (basically the inner field of a list schema element) are not real columns.
      // we can ignore them for the purposes of output nesting info
      if (!cur_schema.is_stub()) {
        // initialize each page within the chunk
        for (size_t p_idx = 0; p_idx < subpass.column_page_count[idx]; p_idx++) {
          // Source file index for the current page.
          auto const src_file_idx =
            pass.chunks[pages[target_page_index + p_idx].chunk_idx].src_file_idx;
          PageNestingInfo* pni =
            &page_nesting_info[nesting_info_index + (p_idx * per_page_nesting_info_size[idx])];

          PageNestingDecodeInfo* nesting_info =
            &page_nesting_decode_info[nesting_info_index +
                                      (p_idx * per_page_nesting_info_size[idx])];

          auto const mapped_src_col_schema =
            _metadata->map_schema_index(src_col_schema, src_file_idx);
          // if we have lists, set our start and end depth remappings
          if (_metadata->get_schema(mapped_src_col_schema, src_file_idx).max_repetition_level > 0) {
            auto remap = depth_remapping.find({src_col_schema, src_file_idx});
            CUDF_EXPECTS(remap != depth_remapping.end(),
                         "Could not find depth remapping for schema");
            std::vector<int> const& rep_depth_remap = (remap->second.first);
            std::vector<int> const& def_depth_remap = (remap->second.second);

            for (size_t m = 0; m < rep_depth_remap.size(); m++) {
              nesting_info[m].start_depth = rep_depth_remap[m];
            }
            for (size_t m = 0; m < def_depth_remap.size(); m++) {
              nesting_info[m].end_depth = def_depth_remap[m];
            }
          }

          // Get the schema from the current input source.
          auto& actual_cur_schema = _metadata->get_schema(
            _metadata->map_schema_index(schema_idx, src_file_idx), src_file_idx);

          // values indexed by output column index
          nesting_info[cur_depth].max_def_level = actual_cur_schema.max_definition_level;
          pni[cur_depth].size                   = 0;
          pni[cur_depth].type =
            to_type_id(actual_cur_schema, _strings_to_categorical, _options.timestamp_type.id());
          pni[cur_depth].nullable = cur_schema.repetition_type == FieldRepetitionType::OPTIONAL;
        }

        // move up the hierarchy
        cur_depth--;
      }

      // next schema
      schema_idx = cur_schema.parent_idx;
      cur_schema = _metadata->get_schema(schema_idx);
    }

    // Offset the page and nesting info indices
    target_page_index += subpass.column_page_count[idx];
    nesting_info_index += (per_page_nesting_info_size[idx] * subpass.column_page_count[idx]);
  }

  // copy nesting info to the device
  page_nesting_info.host_to_device_async(_stream);
  page_nesting_decode_info.host_to_device_async(_stream);
}

void reader_impl::allocate_level_decode_space()
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  auto& pages = subpass.pages;

  // TODO: this could be made smaller if we ignored dictionary pages and pages with no
  // repetition data.
  size_t const per_page_decode_buf_size = LEVEL_DECODE_BUF_SIZE * 2 * pass.level_type_size;
  auto const decode_buf_size            = per_page_decode_buf_size * pages.size();
  subpass.level_decode_data =
    rmm::device_buffer(decode_buf_size, _stream, cudf::get_current_device_resource_ref());

  // distribute the buffers
  auto* buf = static_cast<uint8_t*>(subpass.level_decode_data.data());
  for (size_t idx = 0; idx < pages.size(); idx++) {
    auto& p = pages[idx];

    p.lvl_decode_buf[level_type::DEFINITION] = buf;
    buf += (LEVEL_DECODE_BUF_SIZE * pass.level_type_size);
    p.lvl_decode_buf[level_type::REPETITION] = buf;
    buf += (LEVEL_DECODE_BUF_SIZE * pass.level_type_size);
  }
}

std::pair<bool, std::future<void>> reader_impl::read_column_chunks()
{
  auto const& row_groups_info = _pass_itm_data->row_groups;

  auto& raw_page_data = _pass_itm_data->raw_page_data;
  auto& chunks        = _pass_itm_data->chunks;

  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = _input_columns.size();
  auto const num_chunks        = row_groups_info.size() * num_input_columns;

  // Association between each column chunk and its source
  std::vector<size_type> chunk_source_map(num_chunks);

  // Tracker for eventually deallocating compressed and uncompressed data
  raw_page_data = std::vector<rmm::device_buffer>(num_chunks);

  // Keep track of column chunk file offsets
  std::vector<size_t> column_chunk_offsets(num_chunks);

  // Initialize column chunk information
  size_t total_decompressed_size = 0;
  // TODO: make this respect the pass-wide skip_rows/num_rows instead of the file-wide
  // skip_rows/num_rows
  // auto remaining_rows            = num_rows;
  size_type chunk_count = 0;
  for (auto const& rg : row_groups_info) {
    auto const& row_group       = _metadata->get_row_group(rg.index, rg.source_index);
    auto const row_group_source = rg.source_index;

    // generate ColumnChunkDesc objects for everything to be decoded (all input columns)
    for (size_t i = 0; i < num_input_columns; ++i) {
      auto const& col = _input_columns[i];
      // look up metadata
      auto& col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col.schema_idx);

      column_chunk_offsets[chunk_count] =
        (col_meta.dictionary_page_offset != 0)
          ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
          : col_meta.data_page_offset;

      // Map each column chunk to its column index and its source index
      chunk_source_map[chunk_count] = row_group_source;

      if (col_meta.codec != Compression::UNCOMPRESSED) {
        total_decompressed_size += col_meta.total_uncompressed_size;
      }

      chunk_count++;
    }
  }

  // Read compressed chunk data to device memory
  return {total_decompressed_size > 0,
          read_column_chunks_async(_sources,
                                   raw_page_data,
                                   chunks,
                                   0,
                                   chunks.size(),
                                   column_chunk_offsets,
                                   chunk_source_map,
                                   _stream)};
}

void reader_impl::read_compressed_data()
{
  auto& pass = *_pass_itm_data;

  // This function should never be called if `num_rows == 0`.
  CUDF_EXPECTS(_pass_itm_data->num_rows > 0, "Number of reading rows must not be zero.");

  auto& chunks = pass.chunks;

  auto [has_compressed_data, read_chunks_tasks] = read_column_chunks();
  pass.has_compressed_data                      = has_compressed_data;

  read_chunks_tasks.get();

  // Process dataset chunk pages into output columns
  auto const total_pages = _has_page_index ? count_page_headers_with_pgidx(chunks, _stream)
                                           : count_page_headers(chunks, _stream);
  if (total_pages <= 0) { return; }
  rmm::device_uvector<PageInfo> unsorted_pages(total_pages, _stream);

  // decoding of column/page information
  decode_page_headers(pass, unsorted_pages, _has_page_index, _stream);
  CUDF_EXPECTS(pass.page_offsets.size() - 1 == static_cast<size_t>(_input_columns.size()),
               "Encountered page_offsets / num_columns mismatch");
}

void reader_impl::preprocess_file(read_mode mode)
{
  CUDF_EXPECTS(!_file_preprocessed, "Attempted to preprocess file more than once");

  // if filter is not empty, then create output types as vector and pass for filtering.

  std::vector<data_type> output_dtypes;
  if (_expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });
  }

  std::tie(_file_itm_data.global_skip_rows,
           _file_itm_data.global_num_rows,
           _file_itm_data.row_groups,
           _file_itm_data.num_rows_per_source,
           _file_itm_data.num_input_row_groups,
           _file_itm_data.surviving_row_groups) =
    _metadata->select_row_groups(_sources,
                                 _options.row_group_indices,
                                 _options.skip_rows,
                                 _options.num_rows,
                                 _options.skip_bytes,
                                 _options.num_bytes,
                                 output_dtypes,
                                 _output_column_schemas,
                                 _expr_conv.get_converted_expr(),
                                 _stream);

  CUDF_EXPECTS(
    mode == read_mode::CHUNKED_READ or
      std::cmp_less_equal(_file_itm_data.global_num_rows, std::numeric_limits<size_type>::max()),
    "READ_ALL mode does not support reading number of rows more than cudf's column size limit. "
    "For reading larger number of rows, please use chunked_parquet_reader.",
    std::overflow_error);

  // Inclusive scan the number of rows per source
  if (not _expr_conv.get_converted_expr().has_value() and mode == read_mode::CHUNKED_READ) {
    _file_itm_data.exclusive_sum_num_rows_per_source.resize(
      _file_itm_data.num_rows_per_source.size());
    thrust::inclusive_scan(_file_itm_data.num_rows_per_source.cbegin(),
                           _file_itm_data.num_rows_per_source.cend(),
                           _file_itm_data.exclusive_sum_num_rows_per_source.begin());
  }

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

#if defined(PARQUET_CHUNK_LOGGING)
  printf("==============================================\n");
  setlocale(LC_NUMERIC, "");
  printf("File: skip_rows(%'lu), num_rows(%'lu), input_read_limit(%'lu), output_read_limit(%'lu)\n",
         _file_itm_data.global_skip_rows,
         _file_itm_data.global_num_rows,
         _input_pass_read_limit,
         _output_chunk_read_limit);
  printf("# Row groups: %'lu\n", _file_itm_data.row_groups.size());
  printf("# Input passes: %'lu\n", _file_itm_data.num_passes());
  printf("# Input columns: %'lu\n", _input_columns.size());
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    auto const& schema = _metadata->get_schema(_input_columns[idx].schema_idx);
    auto const type_id = to_type_id(schema, _strings_to_categorical, _options.timestamp_type.id());
    printf("\tC(%'lu, %s): %s\n",
           idx,
           _input_columns[idx].name.c_str(),
           cudf::type_to_name(cudf::data_type{type_id}).c_str());
  }
  printf("# Output columns: %'lu\n", _output_buffers.size());
  for (size_t idx = 0; idx < _output_buffers.size(); idx++) {
    printf("\tC(%'lu): %s\n", idx, cudf::io::detail::type_to_name(_output_buffers[idx]).c_str());
  }
#endif

  _file_preprocessed = true;
}

void reader_impl::generate_list_column_row_counts(is_estimate_row_counts is_estimate_row_counts)
{
  auto& pass = *_pass_itm_data;

  // Computes:
  // Estimated PageInfo::chunk_row (the chunk-relative row index) and PageInfo::num_rows (number of
  // rows in this page) for all pages in the pass. The start_row field in ColumnChunkDesc is the
  // absolute row index for the whole file. chunk_row in PageInfo is relative to the beginning of
  // the chunk. so in the kernels, chunk.start_row + page.chunk_row gives us the absolute row index
  if (is_estimate_row_counts == is_estimate_row_counts::YES) {
    thrust::for_each(rmm::exec_policy(_stream),
                     pass.pages.d_begin(),
                     pass.pages.d_end(),
                     set_list_row_count_estimate{pass.chunks});
    auto key_input  = thrust::make_transform_iterator(pass.pages.d_begin(), get_page_chunk_idx{});
    auto page_input = thrust::make_transform_iterator(pass.pages.d_begin(), get_page_num_rows{});
    thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(_stream),
                                  key_input,
                                  key_input + pass.pages.size(),
                                  page_input,
                                  chunk_row_output_iter{pass.pages.device_ptr()});

    // To compensate for the list row size estimates, force the row count on the last page for each
    // column chunk (each rowgroup) such that it ends on the real known row count. this is so that
    // as we march through the subpasses, we will find that every column cleanly ends up the
    // expected row count at the row group boundary and our split computations work correctly.
    auto iter = thrust::make_counting_iterator(0);
    thrust::for_each(rmm::exec_policy_nosync(_stream),
                     iter,
                     iter + pass.pages.size(),
                     set_final_row_count{pass.pages, pass.chunks});
  } else {
    // If column indexes are available, we don't need to estimate PageInfo::num_rows for lists and
    // can instead translate known PageInfo::chunk_row to PageInfo::num_rows
    thrust::for_each(rmm::exec_policy_nosync(_stream),
                     thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator(pass.pages.size()),
                     compute_page_num_rows_from_chunk_rows{pass.pages, pass.chunks});
  }

  pass.chunks.device_to_host_async(_stream);
  pass.pages.device_to_host_async(_stream);
  _stream.synchronize();
}

void reader_impl::preprocess_subpass_pages(read_mode mode, size_t chunk_read_limit)
{
  CUDF_FUNC_RANGE();

  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  // figure out which kernels to run
  subpass.kernel_mask = get_aggregated_decode_kernel_mask(subpass.pages, _stream);

  // iterate over all input columns and determine if they contain lists.
  // TODO: we could do this once at the file level instead of every time we get in here. the set of
  // columns we are processing does not change over multiple passes/subpasses/output chunks.
  bool has_lists = false;
  for (auto const& input_col : _input_columns) {
    size_t const max_depth = input_col.nesting_depth();

    auto* cols = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // if this has a list parent, we have to get column sizes from the
      // data computed during compute_page_sizes
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) {
        has_lists = true;
        break;
      }
    }
    if (has_lists) { break; }
  }

  // in some cases we will need to do further preprocessing of pages.
  // - if we have lists, the num_rows field in PageInfo will be incorrect coming out of the file
  // - if we are doing a chunked read, we need to compute the size of all string data
  if (has_lists || chunk_read_limit > 0) {
    // computes:
    // PageNestingInfo::num_rows for each page. the true number of rows (taking repetition into
    // account), not just the number of values. PageNestingInfo::size for each level of nesting, for
    // each page.
    //
    // we will be applying a later "trim" pass if skip_rows/num_rows is being used, which can happen
    // if:
    // - user has passed custom row bounds
    // - we will be doing a chunked read
    compute_page_sizes(subpass.pages,
                       pass.chunks,
                       _subpass_page_mask,
                       0,  // 0-max size_t. process all possible rows
                       std::numeric_limits<size_t>::max(),
                       true,  // compute num_rows
                       _pass_itm_data->level_type_size,
                       _stream);
  }

  auto iter = thrust::make_counting_iterator(0);

  // copy our now-correct row counts  back to the base pages stored in the pass.
  // only need to do this if we are not processing the whole pass in one subpass
  if (!subpass.single_subpass) {
    thrust::for_each(rmm::exec_policy_nosync(_stream),
                     iter,
                     iter + subpass.pages.size(),
                     update_pass_num_rows{pass.pages, subpass.pages, subpass.page_src_index});
  }

  // computes:
  // PageInfo::chunk_row (the chunk-relative row index) for all pages in the pass. The start_row
  // field in ColumnChunkDesc is the absolute row index for the whole file. chunk_row in PageInfo is
  // relative to the beginning of the chunk. so in the kernels, chunk.start_row + page.chunk_row
  // gives us the absolute row index
  auto key_input  = thrust::make_transform_iterator(pass.pages.d_begin(), get_page_chunk_idx{});
  auto page_input = thrust::make_transform_iterator(pass.pages.d_begin(), get_page_num_rows{});
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(_stream),
                                key_input,
                                key_input + pass.pages.size(),
                                page_input,
                                chunk_row_output_iter{pass.pages.device_ptr()});

  // copy chunk_row into the subpass pages
  // only need to do this if we are not processing the whole pass in one subpass
  if (!subpass.single_subpass) {
    thrust::for_each(rmm::exec_policy_nosync(_stream),
                     iter,
                     iter + subpass.pages.size(),
                     update_subpass_chunk_row{pass.pages, subpass.pages, subpass.page_src_index});
  }

  // compute string sizes if necessary. if we are doing chunking, we need to know
  // the sizes of all strings so we can properly compute chunk boundaries.
  if ((chunk_read_limit > 0) && (subpass.kernel_mask & STRINGS_MASK)) {
    auto const has_flba =
      std::any_of(pass.chunks.begin(), pass.chunks.end(), [](auto const& chunk) {
        return chunk.physical_type == Type::FIXED_LEN_BYTE_ARRAY and
               is_treat_fixed_length_as_string(chunk.logical_type);
      });
    if (!_has_page_index || has_flba) {
      constexpr bool compute_all_string_sizes = true;
      compute_page_string_sizes_pass1(subpass.pages,
                                      pass.chunks,
                                      _subpass_page_mask,
                                      pass.skip_rows,
                                      pass.num_rows,
                                      subpass.kernel_mask,
                                      compute_all_string_sizes,
                                      _pass_itm_data->level_type_size,
                                      _stream);
    }
    // set str_bytes_all
    thrust::for_each(rmm::exec_policy_nosync(_stream),
                     subpass.pages.device_begin(),
                     subpass.pages.device_end(),
                     set_str_bytes_all{});
  }

  // retrieve pages back
  pass.pages.device_to_host_async(_stream);
  if (!subpass.single_subpass) { subpass.pages.device_to_host_async(_stream); }
  _stream.synchronize();

  // at this point we have an accurate row count so we can compute how many rows we will actually be
  // able to decode for this pass. we will have selected a set of pages for each column in the
  // row group, but not every page will have the same number of rows. so, we can only read as many
  // rows as the smallest batch (by column) we have decompressed.
  size_t first_page_index = 0;
  size_t max_row          = std::numeric_limits<size_t>::max();
  auto const last_pass_row =
    _file_itm_data.input_pass_start_row_count[_file_itm_data._current_input_pass + 1];
  // for each column
  for (size_t idx = 0; idx < subpass.column_page_count.size(); idx++) {
    // compute max row for this column in the subpass
    auto const& last_page  = subpass.pages[first_page_index + (subpass.column_page_count[idx] - 1)];
    auto const& last_chunk = pass.chunks[last_page.chunk_idx];
    auto max_col_row       = static_cast<size_t>(last_chunk.start_row) +
                       static_cast<size_t>(last_page.chunk_row) +
                       static_cast<size_t>(last_page.num_rows);

    // special case.  list rows can span page boundaries, but we can't tell if that is happening
    // here because we have not yet decoded the pages. the very last row starting in the page may
    // not terminate in the page. to handle this, only decode up to the second to last row in the
    // subpass since we know that will safely completed.
    bool const is_list = last_chunk.max_level[level_type::REPETITION] > 0;
    // corner case: only decode up to the second-to-last row, except if this is the last page in the
    // entire pass or if we have the page index. this handles the case where we only have 1 chunk, 1
    // page, and potentially even just 1 row.
    if (is_list and std::cmp_less(max_col_row, last_pass_row) and not _has_page_index) {
      // compute min row for this column in the subpass
      auto const& first_page  = subpass.pages[first_page_index];
      auto const& first_chunk = pass.chunks[first_page.chunk_idx];
      auto const min_col_row =
        static_cast<size_t>(first_chunk.start_row) + static_cast<size_t>(first_page.chunk_row);

      // must have at least 2 rows in the subpass.
      CUDF_EXPECTS((max_col_row - min_col_row) > 1, "Unexpected short subpass");
      max_col_row--;
    }

    max_row = std::min<size_t>(max_row, max_col_row);

    first_page_index += subpass.column_page_count[idx];
  }
  subpass.skip_rows   = pass.skip_rows + pass.processed_rows;
  auto const pass_end = pass.skip_rows + pass.num_rows;
  max_row             = std::min<size_t>(max_row, pass_end);
  CUDF_EXPECTS(max_row > subpass.skip_rows, "Unexpected short subpass", std::underflow_error);
  // Limit the number of rows to read in this subpass to the cudf's column size limit - 1 (for
  // lists)
  subpass.num_rows =
    std::min<size_t>(std::numeric_limits<size_type>::max() - 1, max_row - subpass.skip_rows);

  // now split up the output into chunks as necessary
  compute_output_chunks_for_subpass();
}

void reader_impl::allocate_columns(read_mode mode, size_t skip_rows, size_t num_rows)
{
  CUDF_FUNC_RANGE();

  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  // Should not reach here if there is no page data.
  CUDF_EXPECTS(subpass.pages.size() > 0, "There are no pages present in the subpass");

  // iterate over all input columns and allocate any associated output
  // buffers if they are not part of a list hierarchy. mark down
  // if we have any list columns that need further processing.
  bool has_lists = false;
  // Validity Buffer is a uint32_t pointer
  std::vector<cudf::device_span<cudf::bitmask_type>> nullmask_bufs;

  for (const auto& input_col : _input_columns) {
    size_t const max_depth = input_col.nesting_depth();

    auto* cols = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // if this has a list parent, we have to get column sizes from the
      // data computed during compute_page_sizes
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) {
        has_lists = true;
      }
      // if we haven't already processed this column because it is part of a struct hierarchy
      else if (out_buf.size == 0) {
        // add 1 for the offset if this is a list column
        // we're going to start null mask as all valid and then turn bits off if necessary
        auto const out_buf_size =
          out_buf.type.id() == type_id::LIST && l_idx < max_depth ? num_rows + 1 : num_rows;
        CUDF_EXPECTS(out_buf_size <= std::numeric_limits<cudf::size_type>::max(),
                     "Number of rows exceeds cudf's column size limit",
                     std::overflow_error);
        out_buf.create_with_mask(
          out_buf_size, cudf::mask_state::UNINITIALIZED, false, _stream, _mr);
        nullmask_bufs.emplace_back(
          out_buf.null_mask(),
          cudf::util::round_up_safe(out_buf.null_mask_size(), sizeof(cudf::bitmask_type)) /
            sizeof(cudf::bitmask_type));
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
    rmm::device_uvector<size_t> size_input{num_keys_per_iter, _stream};

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
        get_page_nesting_size{
          d_cols_info.data(), max_depth, subpass.pages.size(), subpass.pages.device_begin()});

      // Manually create a size_t `key_start` compatible counting_transform_iterator.
      auto const reduction_keys =
        thrust::make_transform_iterator(thrust::make_counting_iterator<std::size_t>(key_start),
                                        get_reduction_key{subpass.pages.size()});

      // Find the size of each column
      thrust::reduce_by_key(rmm::exec_policy_nosync(_stream),
                            reduction_keys,
                            reduction_keys + num_keys_this_iter,
                            size_input.cbegin(),
                            thrust::make_discard_iterator(),
                            sizes.d_begin() + (key_start / subpass.pages.size()));

      // For nested hierarchies, compute per-page start offset
      thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(_stream),
                                    reduction_keys,
                                    reduction_keys + num_keys_this_iter,
                                    size_input.cbegin(),
                                    start_offset_output_iterator{subpass.pages.device_begin(),
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
        if ((out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) && out_buf.size == 0) {
          auto buffer_size = sizes[(idx * max_depth) + l_idx];
          // if this is a list column add 1 for non-leaf levels for the terminating offset
          if (out_buf.type.id() == type_id::LIST && l_idx < max_depth) { buffer_size++; }
          CUDF_EXPECTS(buffer_size <= std::numeric_limits<cudf::size_type>::max(),
                       "Number of list column rows exceeds the column size limit. " +
                         std::string((mode == read_mode::CHUNKED_READ)
                                       ? "Consider reducing the `pass_read_limit`."
                                       : ""),
                       std::overflow_error);
          // allocate
          // we're going to start null mask as all valid and then turn bits off if necessary
          out_buf.create_with_mask(
            buffer_size, cudf::mask_state::UNINITIALIZED, false, _stream, _mr);
          nullmask_bufs.emplace_back(
            out_buf.null_mask(),
            cudf::util::round_up_safe(out_buf.null_mask_size(), sizeof(cudf::bitmask_type)) /
              sizeof(cudf::bitmask_type));
        }
      }
    }
  }

  // Need to set null mask bufs to all high bits
  cudf::detail::batched_memset<cudf::bitmask_type>(
    nullmask_bufs, std::numeric_limits<cudf::bitmask_type>::max(), _stream);
}

cudf::detail::host_vector<size_t> reader_impl::calculate_page_string_offsets()
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  auto page_keys = make_page_key_iterator(subpass.pages);

  rmm::device_uvector<size_t> d_col_sizes(_input_columns.size(), _stream);

  // use page_index to fetch page string sizes in the proper order
  auto val_iter = thrust::make_transform_iterator(subpass.pages.device_begin(),
                                                  page_to_string_size{pass.chunks.d_begin()});

  // do scan by key to calculate string offsets for each page
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(_stream),
                                page_keys,
                                page_keys + subpass.pages.size(),
                                val_iter,
                                page_offset_output_iter{subpass.pages.device_ptr()});

  // now sum up page sizes
  rmm::device_uvector<int> reduce_keys(d_col_sizes.size(), _stream);
  thrust::reduce_by_key(rmm::exec_policy_nosync(_stream),
                        page_keys,
                        page_keys + subpass.pages.size(),
                        val_iter,
                        reduce_keys.begin(),
                        d_col_sizes.begin());

  return cudf::detail::make_host_vector(d_col_sizes, _stream);
}

}  // namespace cudf::io::parquet::detail
