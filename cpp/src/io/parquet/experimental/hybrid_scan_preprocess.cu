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

// #include "error.hpp"
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <bitset>
#include <iterator>
#include <limits>
#include <numeric>

namespace cudf::experimental::io::parquet::detail {

void impl::prepare_row_groups(cudf::host_span<std::vector<size_type> const> row_group_indices,
                              cudf::io::parquet_reader_options const& options)
{
  // Save the name to reference converter to extract output filter AST in
  // `preprocess_file()` and `finalize_output()`
  table_metadata metadata;
  populate_metadata(metadata);
  auto expr_conv = named_to_reference_converter(options.get_filter(), metadata);

  std::vector<data_type> output_dtypes;
  if (expr_conv.get_converted_expr().has_value()) {
    std::transform(_output_buffers_template.cbegin(),
                   _output_buffers_template.cend(),
                   std::back_inserter(output_dtypes),
                   [](auto const& col) { return col.type; });

    std::tie(
      _file_itm_data.global_skip_rows, _file_itm_data.global_num_rows, _file_itm_data.row_groups) =
      _metadata->add_row_groups(row_group_indices,
                                options.get_skip_rows(),
                                options.get_num_rows(),
                                output_dtypes,
                                _output_column_schemas,
                                expr_conv.get_converted_expr());

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
}

}  // namespace cudf::experimental::io::parquet::detail