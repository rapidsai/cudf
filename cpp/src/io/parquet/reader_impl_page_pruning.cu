/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/functional.hpp>
#include <cudf/null_mask.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/transform.h>

namespace cudf::io::parquet::detail {

void reader::impl::update_output_bitmasks_for_pruned_pages()
{
  auto const update_null_mask = [&](input_column_info& input_col,
                                    cudf::size_type const start_row,
                                    cudf::size_type const end_row) {
    size_t max_depth = input_col.nesting_depth();
    auto* cols       = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;
      if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) { continue; }
      cudf::set_null_mask(out_buf.null_mask(), start_row, end_row, false, _stream);
      out_buf.null_count() += (end_row - start_row);
    }
  };

  auto const& subpass       = _pass_itm_data->subpass;
  auto const& pages         = subpass->pages;
  auto const& page_validity = subpass->page_validity;
  auto const& chunks        = _pass_itm_data->chunks;
  auto const num_columns    = _input_columns.size();

  for (size_t page_idx = 0; page_idx < pages.size(); page_idx++) {
    if (page_validity[page_idx]) { continue; }
    auto const chunk_idx = pages[page_idx].chunk_idx;
    auto const& chunk    = chunks[chunk_idx];
    auto& input_col      = _input_columns[chunk_idx % num_columns];

    auto const start_row = chunk.start_row + pages[page_idx].chunk_row;
    auto const end_row   = start_row + pages[page_idx].num_rows;
    update_null_mask(input_col, start_row, end_row);
  }
}

}  // namespace cudf::io::parquet::detail
