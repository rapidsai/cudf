/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/types.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <vector>

namespace cudf::io::parquet::detail {

namespace {

// Host-side counterpart of `is_string_col` in `parquet_gpu.hpp`. Kept narrow: for direct
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

// Both PLAIN_DICTIONARY (legacy) and RLE_DICTIONARY are valid encodings for data pages that
// reference a parquet dictionary page.
[[nodiscard]] bool is_dict_data_page_encoding(Encoding enc)
{
  return enc == Encoding::PLAIN_DICTIONARY or enc == Encoding::RLE_DICTIONARY;
}

// Per-input-column eligibility flags. Each column must satisfy all of these conditions to be
// eligible for direct Parquet-dict → DICTIONARY32 transcode.
struct column_eligibility {
  bool has_string_buffer = false;
  bool has_any_chunk     = false;
  bool all_chunks_string = true;
  bool all_pages_dict    = true;

  [[nodiscard]] bool is_eligible() const
  {
    return has_string_buffer and has_any_chunk and all_chunks_string and all_pages_dict;
  }
};

// Classify a chunk against its column's eligibility state.
void update_from_chunk(column_eligibility& e, ColumnChunkDesc const& chunk)
{
  e.has_any_chunk = true;
  if (chunk.max_nesting_depth != 1 or chunk.max_level[level_type::REPETITION] != 0 or
      not is_host_byte_array_string_chunk(chunk) or chunk.num_dict_pages < 1) {
    e.all_chunks_string = false;
  }
}

// Build a STRING keys column covering the dictionary entries of a single chunk of a single input
// column. `begin` points into the pass-wide `string_index_pair` buffer.
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

void reader_impl::prepare_dict_transcode()
{
  CUDF_FUNC_RANGE();

  _dict_transcode_eligible.assign(_input_columns.size(), false);

  if (not _options.try_output_dict_columns) { return; }
  if (_pass_itm_data == nullptr or _pass_itm_data->subpass == nullptr) { return; }

  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  if (pass.chunks.empty() or subpass.pages.size() == 0) { return; }

  // Determine per-input-column eligibility. A column is eligible iff
  //  - the corresponding output buffer is currently typed as STRING (i.e. a flat string column),
  //  - every chunk of that column is a BYTE_ARRAY string chunk with a dictionary page,
  //  - every data page of every chunk of that column uses (PLAIN|RLE)_DICTIONARY encoding,
  //  - the chunk has a flat (non-list, non-nested) schema.
  //
  // We scan host-side `pass.chunks` and `pass.pages` here rather than `subpass.pages` because
  // `subpass.pages` may be a subset. For single-pass single-subpass reads (the only configuration
  // in which `try_output_dict_columns` is supported), `subpass.pages == pass.pages`.
  auto const num_input_cols = _input_columns.size();
  std::vector<column_eligibility> elig(num_input_cols);

  // Seed from the output buffer type: only flat STRING columns have a single leaf buffer whose
  // type is STRING and can be flipped in place.
  std::for_each(
    cuda::counting_iterator<size_t>{0}, cuda::counting_iterator{num_input_cols}, [&](size_t i) {
      auto const& input_col = _input_columns[i];
      if (input_col.nesting_depth() != 1) { return; }
      auto const& out_buf = _output_buffers[input_col.nesting[0]];
      if (out_buf.type.id() == type_id::STRING) { elig[i].has_string_buffer = true; }
    });

  // Fold per-chunk info into the per-column eligibility flags.
  for (auto const& chunk : pass.chunks) {
    auto const col_idx = chunk.src_col_index;
    if (col_idx < 0 or static_cast<size_t>(col_idx) >= num_input_cols) { continue; }
    update_from_chunk(elig[col_idx], chunk);
  }

  // Any non-dictionary data-page encoding disqualifies the whole column. Dictionary pages
  // themselves (PAGEINFO_FLAGS_DICTIONARY) are skipped since they are not data pages.
  for (auto const& page : pass.pages) {
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { continue; }
    auto const chunk_idx = page.chunk_idx;
    if (chunk_idx < 0 or static_cast<size_t>(chunk_idx) >= pass.chunks.size()) { continue; }
    auto const col_idx = pass.chunks[chunk_idx].src_col_index;
    if (col_idx < 0 or static_cast<size_t>(col_idx) >= num_input_cols) { continue; }
    if (not is_dict_data_page_encoding(page.encoding)) { elig[col_idx].all_pages_dict = false; }
  }

  std::transform(
    elig.begin(), elig.end(), _dict_transcode_eligible.begin(), [](column_eligibility const& e) {
      return e.is_eligible();
    });

  auto const num_eligible =
    std::count(_dict_transcode_eligible.begin(), _dict_transcode_eligible.end(), true);
  if (num_eligible == 0) { return; }

  // Flip the output buffer type for eligible columns from STRING → INT32. The subsequent
  // `allocate_columns` call will then allocate an INT32 buffer that the DICT_INT32 kernel can
  // write directly into.
  std::for_each(
    cuda::counting_iterator<size_t>{0}, cuda::counting_iterator{num_input_cols}, [&](size_t i) {
      if (not _dict_transcode_eligible[i]) { return; }
      auto& out_buf = _output_buffers[_input_columns[i].nesting[0]];
      out_buf.type  = data_type{type_id::INT32};
    });

  // Rewrite per-page `kernel_mask` for eligible columns on the host subpass pages from
  // STRING_DICT → DICT_INT32, then H2D so the device pages agree. Only the flat variant is
  // considered here (eligibility requires `max_nesting_depth == 1`).
  bool any_rewritten = false;
  std::for_each(subpass.pages.host_begin(), subpass.pages.host_end(), [&](PageInfo& page) {
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { return; }
    auto const chunk_idx = page.chunk_idx;
    if (chunk_idx < 0 or static_cast<size_t>(chunk_idx) >= pass.chunks.size()) { return; }
    auto const col_idx = pass.chunks[chunk_idx].src_col_index;
    if (col_idx < 0 or static_cast<size_t>(col_idx) >= num_input_cols) { return; }
    if (not _dict_transcode_eligible[col_idx]) { return; }
    if (page.kernel_mask == decode_kernel_mask::STRING_DICT) {
      page.kernel_mask = decode_kernel_mask::DICT_INT32;
      any_rewritten    = true;
    }
  });

  if (not any_rewritten) { return; }

  // Push the rewritten `kernel_mask`s back to device so subsequent decode kernels dispatch
  // correctly. Then refresh the aggregated `subpass.kernel_mask` on the host by re-OR'ing all
  // page kernel masks.
  subpass.pages.host_to_device_async(_stream);
  subpass.kernel_mask = std::transform_reduce(
    subpass.pages.host_begin(),
    subpass.pages.host_end(),
    uint32_t{0},
    std::bit_or<>{},
    [](PageInfo const& page) { return static_cast<uint32_t>(page.kernel_mask); });
  _stream.synchronize();
}

void reader_impl::zero_init_dict_transcoded_index_buffers()
{
  CUDF_FUNC_RANGE();

  if (not _options.try_output_dict_columns) { return; }
  if (_dict_transcode_eligible.empty()) { return; }

  // The `DICT_INT32` kernel only writes to positions with valid definition levels, leaving null
  // slots untouched. Since `allocate_columns` does not zero-initialize fixed-width buffers by
  // default, the INT32 output buffer for a transcoded column may contain uninitialized bytes at
  // null positions. Zero them here so null rows carry a well-defined (valid) index into the
  // dictionary keys -- a requirement for `cudf::dictionary::detail::concatenate` to correctly
  // remap indices below.
  std::for_each(
    cuda::counting_iterator<size_t>{0},
    cuda::counting_iterator{_input_columns.size()},
    [&](size_t i) {
      if (not _dict_transcode_eligible[i]) { return; }
      auto& out_buf = _output_buffers[_input_columns[i].nesting[0]];
      if (out_buf.type.id() != type_id::INT32) { return; }
      if (out_buf.data() == nullptr or out_buf.size == 0) { return; }
      CUDF_CUDA_TRY(cudaMemsetAsync(
        out_buf.data(), 0, static_cast<size_t>(out_buf.size) * sizeof(int32_t), _stream.value()));
    });
}

void reader_impl::assemble_dict_transcoded_columns(
  std::vector<std::unique_ptr<column>>& out_columns)
{
  CUDF_FUNC_RANGE();

  if (not _options.try_output_dict_columns) { return; }
  if (_dict_transcode_eligible.empty()) { return; }
  if (_pass_itm_data == nullptr) { return; }

  auto const& pass = *_pass_itm_data;

  // For each eligible input column, collect its chunks in row-group order, build a per-chunk
  // DICTIONARY32 segment (local 0-based indices + per-chunk keys column), and concatenate.
  //
  // IMPORTANT: Each segment carries row-group-local indices into its own keys column. We do NOT
  // pre-shift indices into a global keyspace, because `cudf::dictionary::detail::concatenate`
  // already re-maps the indices using `compute_children_offsets_fn`. Pre-shifting would cause
  // double-offsetting and out-of-bounds reads in the `dispatch_compute_indices` kernel.
  std::for_each(
    cuda::counting_iterator<size_t>{0},
    cuda::counting_iterator{_input_columns.size()},
    [&](size_t i) {
      if (not _dict_transcode_eligible[i]) { return; }

      // Gather chunk indices for this input column in row-group order.
      std::vector<size_t> chunk_indices;
      chunk_indices.reserve(pass.chunks.size() / std::max<size_t>(_input_columns.size(), 1));
      std::copy_if(cuda::counting_iterator<size_t>{0},
                   cuda::counting_iterator{pass.chunks.size()},
                   std::back_inserter(chunk_indices),
                   [&](size_t c) { return pass.chunks[c].src_col_index == static_cast<int>(i); });
      if (chunk_indices.empty()) { return; }

      // Per-chunk key counts from the dictionary page's `num_input_values`, mirrored back to
      // host when `pass.pages` was copied by `decode_page_headers`.
      std::vector<size_type> chunk_key_counts(chunk_indices.size(), 0);
      std::transform(chunk_indices.begin(),
                     chunk_indices.end(),
                     chunk_key_counts.begin(),
                     [&](size_t chunk_idx) -> size_type {
                       if (pass.chunks[chunk_idx].dict_page == nullptr) { return 0; }
                       for (auto const& page : pass.pages) {
                         if (page.chunk_idx == static_cast<int32_t>(chunk_idx) and
                             (page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) {
                           return static_cast<size_type>(page.num_input_values);
                         }
                       }
                       return size_type{0};
                     });

      // Grab ownership of the decoded INT32 indices column; we slice it into zero-copy
      // per-chunk views to feed into the per-segment dictionary column builders.
      auto& indices_col = out_columns[i];
      CUDF_EXPECTS(indices_col != nullptr and indices_col->type().id() == type_id::INT32,
                   "Expected INT32 indices column for dict-transcoded flat string column");
      auto indices_owner = std::move(indices_col);
      column_view const indices_view{indices_owner->view()};

      // Per-chunk boundaries along the row axis: chunk k occupies rows
      // [chunk_row_offsets[k], chunk_row_offsets[k+1]).
      std::vector<size_type> chunk_row_offsets(chunk_indices.size() + 1, 0);
      std::transform(
        chunk_indices.begin(),
        chunk_indices.end(),
        chunk_row_offsets.begin() + 1,
        [&](size_t chunk_idx) { return static_cast<size_type>(pass.chunks[chunk_idx].num_rows); });
      std::inclusive_scan(
        chunk_row_offsets.begin() + 1, chunk_row_offsets.end(), chunk_row_offsets.begin() + 1);
      CUDF_EXPECTS(chunk_row_offsets.back() == indices_view.size(),
                   "Row counts on pass chunks must sum to the indices column size");

      // Build a DICTIONARY32 segment for every chunk. Each segment carries row-group-local
      // indices into its own keys child; `cudf::detail::concatenate` rewrites the indices
      // against the unified, deduplicated keys. We deep-copy each sliced view into a fresh
      // offset-zero indices column before handing it to `make_dictionary_column`: the
      // `make_dictionary_column(column_view, column_view, ...)` path would otherwise
      // double-apply the slice's offset when constructing the internal indices child.
      std::vector<std::unique_ptr<column>> dict_segments(chunk_indices.size());
      std::transform(cuda::counting_iterator<size_t>{0},
                     cuda::counting_iterator{chunk_indices.size()},
                     dict_segments.begin(),
                     [&](size_t k) {
                       auto const chunk_idx = chunk_indices[k];
                       auto const& chunk    = pass.chunks[chunk_idx];

                       auto const seg_view = cudf::detail::slice(
                         indices_view, chunk_row_offsets[k], chunk_row_offsets[k + 1], _stream);
                       auto seg_indices = std::make_unique<column>(seg_view, _stream, _mr);

                       auto seg_keys = make_keys_column_from_index_pairs(
                         chunk.str_dict_index, chunk_key_counts[k], _stream, _mr);

                       return cudf::make_dictionary_column(
                         std::move(seg_keys), std::move(seg_indices), _stream, _mr);
                     });

      // Concatenate all segments into the final DICTIONARY32 column for this input column.
      // `cudf::dictionary::detail::concatenate` deduplicates + sorts keys and recomputes indices.
      if (dict_segments.size() == 1) {
        out_columns[i] = std::move(dict_segments.front());
      } else {
        std::vector<cudf::column_view> segment_views(dict_segments.size());
        std::transform(
          dict_segments.begin(), dict_segments.end(), segment_views.begin(), [](auto const& seg) {
            return seg->view();
          });
        out_columns[i] = cudf::detail::concatenate(segment_views, _stream, _mr);
      }
    });
}

}  // namespace cudf::io::parquet::detail
