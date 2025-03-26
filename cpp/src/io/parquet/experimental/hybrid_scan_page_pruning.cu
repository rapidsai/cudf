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

#include "cudf/detail/utilities/host_vector.hpp"
#include "cudf/detail/utilities/vector_factories.hpp"
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

void impl::set_page_validity(cudf::host_span<std::vector<bool> const> data_page_validity)
{
  CUDF_EXPECTS(_file_itm_data._current_input_pass < _file_itm_data.num_passes(), "Invalid pass");

  auto const& chunks = _pass_itm_data->chunks;
  _page_validity     = cudf::detail::make_host_vector<bool>(_pass_itm_data->pages.size(), _stream);

  auto page_idx = 0;
  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(chunks.size()),
                [&](auto chunk_idx) {
                  if (chunks[chunk_idx].num_dict_pages > 0) {
                    _page_validity[page_idx] = true;
                    page_idx++;
                  }
                  CUDF_EXPECTS(
                    data_page_validity[chunk_idx].size() == chunks[chunk_idx].num_data_pages,
                    "Mismatched data page validity size");
                  std::copy(data_page_validity[chunk_idx].begin(),
                            data_page_validity[chunk_idx].end(),
                            _page_validity.begin() + page_idx);
                  page_idx += data_page_validity[chunk_idx].size();
                });
}

void impl::update_output_bitmasks_for_pruned_pages()
{
  auto const update_null_mask = [&](input_column_info& input_col,
                                    cudf::size_type const start_row,
                                    cudf::size_type const end_row) {
    size_t max_depth = input_col.nesting_depth();
    auto* cols       = &_output_buffers;
    for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;
      if (out_buf.user_data &
          cudf::io::parquet::detail::PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) {
        continue;
      }
      cudf::set_null_mask(out_buf.null_mask(), start_row, end_row, false, _stream);
      out_buf.null_count() += (end_row - start_row);
    }
  };

  auto const& pages      = _pass_itm_data->pages;
  auto const& chunks     = _pass_itm_data->chunks;
  auto const num_columns = _input_columns.size();

  auto page_idx = 0;
  for (page_idx = 0; page_idx < pages.size(); page_idx++) {
    if (_page_validity[page_idx]) { continue; }
    auto const chunk_idx = pages[page_idx].chunk_idx;
    auto const& chunk    = chunks[chunk_idx];
    auto& input_col      = _input_columns[chunk_idx % num_columns];

    auto const start_row = chunk.start_row + pages[page_idx].chunk_row;
    auto const end_row   = start_row + pages[page_idx].num_rows;
    update_null_mask(input_col, start_row, end_row);
  }
}

}  // namespace cudf::experimental::io::parquet::detail