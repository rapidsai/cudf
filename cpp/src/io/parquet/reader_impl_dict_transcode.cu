/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @brief Host-side check for whether a column chunk decodes to a plain string column.
 *
 * Host-side counterpart of `is_string_col` in `parquet_gpu.hpp`. Kept narrow: for direct
 * Parquet-dict → DICTIONARY32 transcode we only accept pure BYTE_ARRAY columns without a
 * DECIMAL logical type and without the strings-to-categorical flag. FIXED_LEN_BYTE_ARRAY is
 * deliberately excluded because it is typically a binary payload.
 *
 * @param chunk The column chunk descriptor to classify
 * @return True if the chunk is a plain BYTE_ARRAY string chunk eligible for transcode
 */
[[nodiscard]] bool is_host_byte_array_string_chunk(ColumnChunkDesc const& chunk)
{
  if (chunk.physical_type != Type::BYTE_ARRAY) { return false; }
  if (chunk.is_strings_to_cat) { return false; }
  if (chunk.logical_type.has_value() and chunk.logical_type->type == LogicalType::DECIMAL) {
    return false;
  }
  return true;
}

/**
 * @brief Whether a data-page encoding references a parquet dictionary page.
 *
 * Both PLAIN_DICTIONARY (legacy) and RLE_DICTIONARY are valid encodings for data pages that
 * reference a parquet dictionary page.
 *
 * @param enc The data-page encoding to test
 * @return True if the encoding is a dictionary data-page encoding
 */
[[nodiscard]] bool is_dict_data_page_encoding(Encoding enc)
{
  return enc == Encoding::PLAIN_DICTIONARY or enc == Encoding::RLE_DICTIONARY;
}

/**
 * @brief Per-input-column eligibility flags for Parquet-dict → DICTIONARY32 transcode.
 *
 * Each column must satisfy all of these conditions to be eligible for direct transcode.
 */
struct column_eligibility {
  bool has_string_buffer = false;  ///< Output buffer is currently typed as STRING
  bool has_any_chunk     = false;  ///< At least one chunk was seen for this column
  bool all_chunks_string = true;   ///< Every chunk is a flat BYTE_ARRAY string chunk with a dict
  bool all_pages_dict    = true;   ///< Every data page uses a dictionary encoding

  /**
   * @brief Whether the column satisfies every transcode-eligibility condition.
   *
   * @return True if the column is eligible for direct DICTIONARY32 transcode
   */
  [[nodiscard]] bool is_eligible() const
  {
    return has_string_buffer and has_any_chunk and all_chunks_string and all_pages_dict;
  }
};

/**
 * @brief Fold a single chunk's properties into its column's eligibility state.
 *
 * @param e The per-column eligibility state to update in place
 * @param chunk The column chunk descriptor to classify
 */
void update_from_chunk(column_eligibility& e, ColumnChunkDesc const& chunk)
{
  e.has_any_chunk = true;
  if (chunk.max_nesting_depth != 1 or chunk.max_level[level_type::REPETITION] != 0 or
      not is_host_byte_array_string_chunk(chunk) or chunk.num_dict_pages < 1) {
    e.all_chunks_string = false;
  }
}

/**
 * @brief Compute per-input-column eligibility for Parquet-dict → DICTIONARY32 transcode.
 *
 * A column is eligible iff
 *  - the corresponding output buffer is currently typed as STRING (i.e. a flat string column),
 *  - every chunk of that column is a BYTE_ARRAY string chunk with a dictionary page,
 *  - every data page of every chunk of that column uses (PLAIN|RLE)_DICTIONARY encoding,
 *  - the chunk has a flat (non-list, non-nested) schema.
 *
 * We scan host-side `pass.chunks` and `pass.pages` here rather than `subpass.pages` because
 * `subpass.pages` may be a subset. For single-pass single-subpass reads (the only configuration
 * in which `try_output_dict_columns` is supported), `subpass.pages == pass.pages`.
 *
 * @param pass The pass intermediate data holding host-side chunks and pages
 * @param input_columns The reader's input column descriptors
 * @param output_buffers The output column buffers (used to detect flat STRING columns)
 * @return A vector of per-input-column eligibility records, indexed by input column
 */
[[nodiscard]] std::vector<column_eligibility> compute_dict_transcode_eligibility(
  pass_intermediate_data const& pass,
  std::vector<input_column_info> const& input_columns,
  std::vector<cudf::io::detail::inline_column_buffer> const& output_buffers)
{
  auto const num_input_cols = input_columns.size();
  std::vector<column_eligibility> elig(num_input_cols);

  // Check if the output buffer is a flat string column
  std::for_each(
    cuda::counting_iterator<size_t>{0}, cuda::counting_iterator{num_input_cols}, [&](size_t i) {
      auto const& input_col = input_columns[i];
      if (input_col.nesting_depth() != 1) { return; }
      if (output_buffers[input_col.nesting[0]].type.id() == type_id::STRING) {
        elig[i].has_string_buffer = true;
      }
    });

  // Fold per-chunk info into the per-column eligibility flags.
  for (auto const& chunk : pass.chunks) {
    auto const col_idx = chunk.src_col_index;
    update_from_chunk(elig[col_idx], chunk);
  }

  // Any non-dictionary data-page encoding disqualifies the whole column. Dictionary pages
  // themselves (PAGEINFO_FLAGS_DICTIONARY) are skipped since they are not data pages.
  for (auto const& page : pass.pages) {
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { continue; }
    auto const chunk_idx = page.chunk_idx;
    auto const col_idx   = pass.chunks[chunk_idx].src_col_index;
    if (not is_dict_data_page_encoding(page.encoding)) { elig[col_idx].all_pages_dict = false; }
  }

  return elig;
}

/**
 * @brief Build a STRING keys column from a chunk's dictionary entries.
 *
 * Builds a STRING keys column covering the dictionary entries of a single chunk of a single input
 * column. `begin` points into the pass-wide `string_index_pair` buffer.
 *
 * @param begin Pointer to the first `string_index_pair` entry for this chunk's dictionary
 * @param entry_count Number of dictionary entries (keys) for this chunk
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's memory
 * @return A STRING column holding this chunk's dictionary keys (empty if `entry_count <= 0`)
 */
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

bool reader_impl::prepare_dict_transcode()
{
  CUDF_FUNC_RANGE();

  _dict_transcode_eligible.assign(_input_columns.size(), false);

  if (not _options.try_output_dict_columns) { return false; }
  if (_pass_itm_data == nullptr or _pass_itm_data->subpass == nullptr) { return false; }

  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  if (pass.chunks.empty() or subpass.pages.size() == 0) { return false; }

  auto const elig = compute_dict_transcode_eligibility(pass, _input_columns, _output_buffers);
  std::transform(
    elig.begin(), elig.end(), _dict_transcode_eligible.begin(), [](column_eligibility const& e) {
      return e.is_eligible();
    });

  auto const num_eligible =
    std::count(_dict_transcode_eligible.begin(), _dict_transcode_eligible.end(), true);
  if (num_eligible == 0) { return false; }

  auto const num_input_cols = _input_columns.size();

  // Change the output buffer type for eligible columns from STRING → INT32. The subsequent
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
    auto const col_idx   = pass.chunks[chunk_idx].src_col_index;
    if (not _dict_transcode_eligible[col_idx]) { return; }
    if (page.kernel_mask == decode_kernel_mask::STRING_DICT) {
      page.kernel_mask = decode_kernel_mask::DICT_INT32;
      any_rewritten    = true;
    }
  });

  if (not any_rewritten) { return false; }

  // Push the rewritten `kernel_mask`s back to device so subsequent decode kernels dispatch
  // correctly. The copy is enqueued on `_stream`, so no explicit synchronization is required. The
  // host source buffer (`subpass.pages`) is owned by the subpass and is neither freed nor
  // re-mutated before the copy completes.
  subpass.pages.host_to_device_async(_stream);
  subpass.kernel_mask = std::transform_reduce(
    subpass.pages.host_begin(),
    subpass.pages.host_end(),
    uint32_t{0},
    std::bit_or<>{},
    [](PageInfo const& page) { return static_cast<uint32_t>(page.kernel_mask); });
  return true;
}

void reader_impl::zero_init_dict_transcoded_index_buffers()
{
  CUDF_FUNC_RANGE();

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

      // Grab ownership of the decoded INT32 indices column. Its buffer is shared (aliased) by
      // every per-chunk DICTIONARY32 view below via the parent view's offset/size, so it must
      // stay alive until the per-column concatenate/assembly completes.
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

      // Build a DICTIONARY32 *view* for every chunk without copying the decoded indices. Each
      // view's keys child is this chunk's own STRING keys column (which must be materialized from
      // the parquet dictionary page), while its indices child aliases the shared, already-decoded
      // INT32 buffer. We select each chunk's row range via the parent dictionary view's
      // `offset`/`size` rather than slicing the indices child: `get_indices_annotated()` rebuilds
      // the indices view from the child's `head()` plus the parent's offset/size, so a sliced
      // child (carrying its own offset) would be ignored. `cudf::detail::concatenate` then
      // rewrites the row-group-local indices against the unified, deduplicated keys.
      std::vector<std::unique_ptr<column>> seg_keys_owners(chunk_indices.size());
      std::vector<column_view> dict_segment_views(chunk_indices.size());
      std::transform(cuda::counting_iterator<size_t>{0},
                     cuda::counting_iterator{chunk_indices.size()},
                     dict_segment_views.begin(),
                     [&](size_t k) {
                       auto const chunk_idx = chunk_indices[k];
                       auto const& chunk    = pass.chunks[chunk_idx];

                       seg_keys_owners[k] = make_keys_column_from_index_pairs(
                         chunk.str_dict_index, chunk_key_counts[k], _stream, _mr);

                       auto const seg_rows = chunk_row_offsets[k + 1] - chunk_row_offsets[k];
                       return column_view{data_type{type_id::DICTIONARY32},
                                          seg_rows,
                                          nullptr,               // dictionary parent holds no data
                                          nullptr,               // non-nullable transcode path
                                          0,                     // null count
                                          chunk_row_offsets[k],  // reslices shared indices child
                                          {indices_view, seg_keys_owners[k]->view()}};
                     });

      // Materialize the final DICTIONARY32 column for this input column.
      if (dict_segment_views.size() == 1) {
        // Single row group: the parquet dictionary page keys are already unique, so no dedup is
        // needed. Take ownership of the decoded INT32 indices buffer directly (zero copy).
        out_columns[i] = cudf::make_dictionary_column(
          std::move(seg_keys_owners.front()), std::move(indices_owner), _stream, _mr);
      } else {
        // `cudf::detail::concatenate` deduplicates + sorts keys and recomputes indices.
        out_columns[i] = cudf::detail::concatenate(dict_segment_views, _stream, _mr);
      }
    });
}

}  // namespace cudf::io::parquet::detail
