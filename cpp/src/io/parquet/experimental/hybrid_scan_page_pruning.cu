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
#include "io/parquet/parquet_gpu.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>

namespace cudf::experimental::io::parquet::detail {

using input_column_info    = cudf::io::parquet::detail::input_column_info;
using inline_column_buffer = cudf::io::detail::inline_column_buffer;

namespace {

void update_nullmask(input_column_info& input_col,
                     std::vector<inline_column_buffer>& output_buffers,
                     cudf::size_type const start_row,
                     cudf::size_type const end_row,
                     rmm::cuda_stream_view stream)
{
  size_t max_depth = input_col.nesting_depth();
  auto* cols       = &output_buffers;
  for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
    auto& out_buf = (*cols)[input_col.nesting[l_idx]];
    cols          = &out_buf.children;
    if (out_buf.user_data & cudf::io::parquet::detail::PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) {
      continue;
    }
    cudf::set_null_mask(out_buf.null_mask(), start_row, end_row, false, stream);
    out_buf.null_count() += (end_row - start_row);
  }
}

}  // namespace

void impl::set_page_validity(cudf::host_span<std::vector<bool> const> data_page_validity)
{
  CUDF_EXPECTS(_file_itm_data._current_input_pass < _file_itm_data.num_passes(), "Invalid pass");

  auto const& pass   = _pass_itm_data;
  auto const& chunks = pass->chunks;
  _page_validity.reserve(pass->pages.size());
  auto const num_columns = _input_columns.size();

  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(_input_columns.size()),
    [&](auto col_idx) {
      auto const& col_page_validity = data_page_validity[col_idx];
      size_t num_inserted_pages     = 0;
      for (size_t chunk_idx = col_idx; chunk_idx < chunks.size(); chunk_idx += num_columns) {
        if (chunks[chunk_idx].num_dict_pages > 0) { _page_validity.emplace_back(true); }
        CUDF_EXPECTS(
          col_page_validity.size() >= num_inserted_pages + chunks[chunk_idx].num_data_pages,
          "Encountered unavailable validity for data pages");
        _page_validity.insert(
          _page_validity.end(),
          col_page_validity.begin() + num_inserted_pages,
          col_page_validity.begin() + num_inserted_pages + chunks[chunk_idx].num_data_pages);
        num_inserted_pages += chunks[chunk_idx].num_data_pages;
      }
      CUDF_EXPECTS(num_inserted_pages == col_page_validity.size(),
                   "Encountered mismatch in data pages and validity sizes");
    });
}

void impl::update_output_nullmasks_for_pruned_pages()
{
  auto const& pages      = _pass_itm_data->pages;
  auto const& chunks     = _pass_itm_data->chunks;
  auto const num_columns = _input_columns.size();

  CUDF_EXPECTS(pages.size() == _page_validity.size(), "Page validity size mismatch");

  thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(pages.begin(), _page_validity.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(pages.end(), _page_validity.end())),
    [&](auto const& page_and_validity_pair) {
      auto const& page      = thrust::get<0>(page_and_validity_pair);
      auto const page_valid = thrust::get<1>(page_and_validity_pair);

      // Return if the page is valid
      if (page_valid) { return; }

      // Update nullmask for the current page
      auto const chunk_idx = page.chunk_idx;
      auto& input_col      = _input_columns[chunk_idx % num_columns];

      auto const start_row = chunks[chunk_idx].start_row + page.chunk_row;
      auto const end_row   = start_row + page.num_rows;

      update_nullmask(input_col, _output_buffers, start_row, end_row, _stream);
    });
}

}  // namespace cudf::experimental::io::parquet::detail
