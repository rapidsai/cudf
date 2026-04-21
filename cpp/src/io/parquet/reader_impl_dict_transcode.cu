/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <algorithm>
#include <vector>

namespace cudf::io::parquet::detail {

namespace {

// Host-side counterpart of `is_string_col` in parquet_gpu.hpp. Kept narrow: for direct
// Parquet-dict → DICTIONARY32 transcode we only accept pure BYTE_ARRAY columns without a
// DECIMAL logical type and without the strings-to-categorical flag. FIXED_LEN_BYTE_ARRAY is
// deliberately excluded because it is typically a binary payload.
[[nodiscard]] bool is_host_byte_array_string_chunk(ColumnChunkDesc const& chunk)
{
  if (chunk.physical_type != Type::BYTE_ARRAY) { return false; }
  if (chunk.is_strings_to_cat) { return false; }
  if (chunk.logical_type.has_value() and chunk.logical_type->type == LogicalType::DECIMAL) {
    return false;
  }
  return true;
}

// Is the given page encoding a dictionary-indices encoding? Both PLAIN_DICTIONARY (legacy) and
// RLE_DICTIONARY are valid encodings for data pages that reference a parquet dictionary page.
[[nodiscard]] bool is_dict_data_page_encoding(Encoding enc)
{
  return enc == Encoding::PLAIN_DICTIONARY or enc == Encoding::RLE_DICTIONARY;
}

}  // namespace

void reader_impl::prepare_dict_transcode()
{
  CUDF_FUNC_RANGE();

  _dict_transcode_eligible.assign(_input_columns.size(), false);

  if (not _options.try_output_dict_columns) { return; }
  if (_pass_itm_data == nullptr or _pass_itm_data->subpass == nullptr) { return; }

  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  if (pass.chunks.empty() or subpass.pages.size() == 0) { return; }

  // Step 1: determine per-input-column eligibility. A column is eligible iff
  //  - the corresponding output buffer is currently typed as STRING (i.e. a flat string column),
  //  - every chunk of that column is a BYTE_ARRAY string chunk with a dictionary page,
  //  - every data page of every chunk of that column uses (PLAIN|RLE)_DICTIONARY encoding,
  //  - the chunk has a flat (non-list, non-nested) schema.
  //
  // We scan host-side pass.chunks and pass.pages here rather than subpass.pages because
  // subpass.pages may be a subset. For single-pass single-subpass reads (the only configuration
  // in which try_output_dict_columns is supported), subpass.pages == pass.pages.
  auto const num_input_cols = _input_columns.size();

  std::vector<bool> col_has_string_buffer(num_input_cols, false);
  std::vector<bool> col_all_chunks_string = std::vector<bool>(num_input_cols, true);
  std::vector<bool> col_has_any_chunk     = std::vector<bool>(num_input_cols, false);
  std::vector<bool> col_all_pages_dict    = std::vector<bool>(num_input_cols, true);

  for (size_t i = 0; i < num_input_cols; ++i) {
    auto const& input_col = _input_columns[i];
    // Flat columns have nesting_depth == 1, and the root output buffer is the leaf.
    if (input_col.nesting_depth() != 1) { continue; }
    auto const& out_buf = _output_buffers[input_col.nesting[0]];
    if (out_buf.type.id() == type_id::STRING) { col_has_string_buffer[i] = true; }
  }

  for (size_t c = 0; c < pass.chunks.size(); ++c) {
    auto const& chunk   = pass.chunks[c];
    auto const col_idx  = chunk.src_col_index;
    if (col_idx < 0 or static_cast<size_t>(col_idx) >= num_input_cols) { continue; }
    col_has_any_chunk[col_idx] = true;
    if (chunk.max_nesting_depth != 1 or chunk.max_level[level_type::REPETITION] != 0 or
        not is_host_byte_array_string_chunk(chunk) or chunk.num_dict_pages < 1) {
      col_all_chunks_string[col_idx] = false;
    }
  }

  for (auto const& page : pass.pages) {
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { continue; }
    auto const chunk_idx = page.chunk_idx;
    if (chunk_idx < 0 or static_cast<size_t>(chunk_idx) >= pass.chunks.size()) { continue; }
    auto const col_idx = pass.chunks[chunk_idx].src_col_index;
    if (col_idx < 0 or static_cast<size_t>(col_idx) >= num_input_cols) { continue; }
    if (not is_dict_data_page_encoding(page.encoding)) {
      col_all_pages_dict[col_idx] = false;
    }
  }

  for (size_t i = 0; i < num_input_cols; ++i) {
    _dict_transcode_eligible[i] = col_has_string_buffer[i] and col_has_any_chunk[i] and
                                  col_all_chunks_string[i] and col_all_pages_dict[i];
  }

  auto const num_eligible =
    std::count(_dict_transcode_eligible.begin(), _dict_transcode_eligible.end(), true);
  if (num_eligible == 0) { return; }

  // Step 2: flip the output buffer type for eligible columns from STRING → INT32. The subsequent
  // `allocate_columns` call will then allocate an INT32 buffer that the DICT_INT32 kernel can
  // write directly into.
  for (size_t i = 0; i < num_input_cols; ++i) {
    if (not _dict_transcode_eligible[i]) { continue; }
    auto& out_buf = _output_buffers[_input_columns[i].nesting[0]];
    out_buf.type  = data_type{type_id::INT32};
    // Leaf flat columns have no children; nothing more to adjust.
  }

  // Step 3: rewrite per-page kernel_mask for eligible columns on the host subpass pages from
  // STRING_DICT → DICT_INT32, then H2D so the device pages agree. Only the flat variant is
  // considered here (eligibility requires `max_nesting_depth == 1`).
  bool any_rewritten = false;
  for (size_t p = 0; p < subpass.pages.size(); ++p) {
    auto& page = subpass.pages[p];
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { continue; }
    auto const chunk_idx = page.chunk_idx;
    if (chunk_idx < 0 or static_cast<size_t>(chunk_idx) >= pass.chunks.size()) { continue; }
    auto const col_idx = pass.chunks[chunk_idx].src_col_index;
    if (col_idx < 0 or static_cast<size_t>(col_idx) >= num_input_cols) { continue; }
    if (not _dict_transcode_eligible[col_idx]) { continue; }
    if (page.kernel_mask == decode_kernel_mask::STRING_DICT) {
      page.kernel_mask = decode_kernel_mask::DICT_INT32;
      any_rewritten    = true;
    }
  }

  if (any_rewritten) {
    // Push the rewritten kernel_masks back to device so subsequent decode kernels dispatch
    // correctly. Then refresh the aggregated subpass.kernel_mask on the host by re-OR'ing all
    // page kernel_masks.
    subpass.pages.host_to_device_async(_stream);
    uint32_t refreshed = 0;
    for (size_t p = 0; p < subpass.pages.size(); ++p) {
      refreshed |= static_cast<uint32_t>(subpass.pages[p].kernel_mask);
    }
    subpass.kernel_mask = refreshed;
    _stream.synchronize();
  }
}

void reader_impl::zero_init_dict_transcoded_index_buffers()
{
  if (not _options.try_output_dict_columns) { return; }
  if (_dict_transcode_eligible.empty()) { return; }

  // The `DICT_INT32` kernel only writes to positions with valid definition levels, leaving null
  // slots untouched. Since `allocate_columns` uses `memset_data=false` by default, the INT32
  // output buffer for a transcoded column may contain uninitialized bytes at null positions.
  // Zero them here so null rows carry a well-defined (valid) index into the dictionary keys.
  for (size_t i = 0; i < _input_columns.size(); ++i) {
    if (not _dict_transcode_eligible[i]) { continue; }
    auto const& input_col = _input_columns[i];
    auto& out_buf         = _output_buffers[input_col.nesting[0]];
    if (out_buf.type.id() != type_id::INT32) { continue; }
    if (out_buf.data() == nullptr or out_buf.size == 0) { continue; }
    CUDF_CUDA_TRY(cudaMemsetAsync(
      out_buf.data(), 0, static_cast<size_t>(out_buf.size) * sizeof(int32_t), _stream.value()));
  }
}

namespace {

// Build a STRING keys column covering the dictionary entries of a contiguous range of chunks of a
// single input column. `str_dict_index` is the device-resident pointer to the pass-wide
// `string_index_pair` buffer. `entry_count` is the total number of entries contributed by the
// range. Because the pass's str_dict_index buffer already stores pairs in chunk-index order with
// packed offsets, and our caller passes a range covering a contiguous sub-slice for one column,
// we can simply wrap the pointer in a span and hand it to the strings factory.
[[nodiscard]] std::unique_ptr<column> make_keys_column_from_index_pairs(
  string_index_pair const* begin,
  size_type entry_count,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (entry_count <= 0) { return cudf::make_empty_column(data_type{type_id::STRING}); }
  return cudf::strings::detail::make_strings_column(begin, begin + entry_count, stream, mr);
}

}  // namespace

void reader_impl::assemble_dict_transcoded_columns(
  std::vector<std::unique_ptr<column>>& out_columns)
{
  CUDF_FUNC_RANGE();

  if (not _options.try_output_dict_columns) { return; }
  if (_dict_transcode_eligible.empty()) { return; }
  if (_pass_itm_data == nullptr) { return; }

  auto& pass = *_pass_itm_data;

  // For each eligible input column, collect its chunks in row-group order, build a per-chunk
  // DICTIONARY32 segment (local 0-based indices + per-chunk keys column), and concatenate.
  //
  // IMPORTANT: Each segment carries row-group-local indices into its own keys column. We do NOT
  // pre-shift indices into a global keyspace, because `cudf::concatenate` on dictionary columns
  // (via `cudf::dictionary::detail::concatenate`) already re-maps the indices using
  // `compute_children_offsets_fn`. Pre-shifting would cause double-offsetting and out-of-bounds
  // reads in the `dispatch_compute_indices` kernel.
  for (size_t i = 0; i < _input_columns.size(); ++i) {
    if (not _dict_transcode_eligible[i]) { continue; }

    // Gather the ordered list of chunk indices belonging to this input column.
    std::vector<size_t> chunk_indices;
    chunk_indices.reserve(pass.chunks.size() / std::max<size_t>(_input_columns.size(), 1));
    for (size_t c = 0; c < pass.chunks.size(); ++c) {
      if (pass.chunks[c].src_col_index == static_cast<int>(i)) { chunk_indices.push_back(c); }
    }
    if (chunk_indices.empty()) { continue; }

    // Per-chunk key counts derived from the dictionary page's num_input_values, mirrored back to
    // the host when `pass.pages` was copied by `decode_page_headers`.
    std::vector<size_type> chunk_key_counts(chunk_indices.size(), 0);
    for (size_t k = 0; k < chunk_indices.size(); ++k) {
      auto const& chunk = pass.chunks[chunk_indices[k]];
      if (chunk.dict_page != nullptr) {
        for (auto const& page : pass.pages) {
          if (page.chunk_idx == static_cast<int32_t>(chunk_indices[k]) and
              (page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) {
            chunk_key_counts[k] = static_cast<size_type>(page.num_input_values);
            break;
          }
        }
      }
    }

    // Grab ownership of the decoded INT32 indices column and its raw device pointer so we can
    // carve per-chunk slices out of it without additional synchronization.
    auto& indices_col = out_columns[i];
    CUDF_EXPECTS(indices_col != nullptr and indices_col->type().id() == type_id::INT32,
                 "Expected INT32 indices column for dict-transcoded flat string column");

    std::vector<size_type> chunk_row_offsets(chunk_indices.size() + 1, 0);
    for (size_t k = 0; k < chunk_indices.size(); ++k) {
      chunk_row_offsets[k + 1] =
        chunk_row_offsets[k] + static_cast<size_type>(pass.chunks[chunk_indices[k]].num_rows);
    }

    auto indices_contents = indices_col->release();
    auto* indices_data    = static_cast<int32_t const*>(indices_contents.data->data());
    auto const indices_size =
      static_cast<size_type>(indices_contents.data->size() / sizeof(int32_t));
    CUDF_EXPECTS(indices_size == chunk_row_offsets.back(),
                 "Row counts on pass chunks must sum to the indices column size");

    std::vector<std::unique_ptr<column>> dict_segments;
    dict_segments.reserve(chunk_indices.size());

    for (size_t k = 0; k < chunk_indices.size(); ++k) {
      auto const& chunk      = pass.chunks[chunk_indices[k]];
      auto const row_begin   = chunk_row_offsets[k];
      auto const row_end     = chunk_row_offsets[k + 1];
      auto const key_count   = chunk_key_counts[k];
      auto const chunk_nrows = row_end - row_begin;

      // Copy this chunk's slice of (unshifted, local-to-chunk) INT32 indices into its own buffer.
      rmm::device_buffer seg_data(chunk_nrows * sizeof(int32_t), _stream, _mr);
      if (chunk_nrows > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(seg_data.data(),
                                      indices_data + row_begin,
                                      chunk_nrows * sizeof(int32_t),
                                      cudaMemcpyDeviceToDevice,
                                      _stream.value()));
      }

      // Slice the null mask into this chunk's range.
      rmm::device_buffer seg_null_mask{};
      size_type seg_null_count = 0;
      if (indices_contents.null_mask != nullptr and indices_contents.null_mask->size() > 0) {
        auto const* src_mask_ptr =
          static_cast<bitmask_type const*>(indices_contents.null_mask->data());
        seg_null_mask  = cudf::detail::copy_bitmask(src_mask_ptr, row_begin, row_end, _stream, _mr);
        seg_null_count = cudf::null_count(src_mask_ptr, row_begin, row_end, _stream);
      }

      auto seg_indices = std::make_unique<column>(data_type{type_id::INT32},
                                                  chunk_nrows,
                                                  std::move(seg_data),
                                                  std::move(seg_null_mask),
                                                  seg_null_count);

      auto seg_keys =
        make_keys_column_from_index_pairs(chunk.str_dict_index, key_count, _stream, _mr);

      // Assemble a DICTIONARY32 column for this chunk segment. Indices are local 0-based; the
      // subsequent `cudf::concatenate` will rewrite them against the unified, deduplicated keys.
      auto seg_dict =
        cudf::make_dictionary_column(std::move(seg_keys), std::move(seg_indices), _stream, _mr);
      dict_segments.emplace_back(std::move(seg_dict));
    }

    // Concatenate all segments into the final DICTIONARY32 column for this input column.
    // `cudf::dictionary::detail::concatenate` deduplicates + sorts keys and recomputes indices.
    if (dict_segments.size() == 1) {
      out_columns[i] = std::move(dict_segments.front());
    } else {
      std::vector<cudf::column_view> segment_views;
      segment_views.reserve(dict_segments.size());
      for (auto const& seg : dict_segments) {
        segment_views.emplace_back(seg->view());
      }
      out_columns[i] = cudf::concatenate(segment_views, _stream, _mr);
    }
  }
}

}  // namespace cudf::io::parquet::detail
