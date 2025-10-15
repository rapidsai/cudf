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
#include "io/parquet/stats_filter_helpers.hpp"
#include "page_index_filter_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/logger.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <algorithm>
#include <limits>

namespace cudf::io::parquet::experimental::detail {

using metadata_base     = parquet::detail::metadata;
using stats_caster_base = parquet::detail::stats_caster_base;
using string_index_pair = parquet::detail::string_index_pair;

namespace {

/**
 * @brief Converts page-level statistics of a column to 2 device columns - min, max values. Each
 * column has number of rows equal to the total rows in all row groups.
 */
struct page_stats_caster : public stats_caster_base {
  cudf::size_type total_rows;
  cudf::host_span<metadata_base const> per_file_metadata;
  cudf::host_span<std::vector<size_type> const> row_group_indices;

  page_stats_caster(size_type total_rows,
                    cudf::host_span<metadata_base const> per_file_metadata,
                    cudf::host_span<std::vector<size_type> const> row_group_indices)
    : total_rows{total_rows},
      per_file_metadata{per_file_metadata},
      row_group_indices{row_group_indices}
  {
  }

  /**
   * @brief Transforms a page-level stats column to a row-level stats column for non-string types
   *
   * @tparam T The data type of the column - must be non-compound
   * @param column Mutable view of input page-level device column
   * @param page_nullmask Host nullmask of the input page-level column
   * @param page_indices Device vector containing the page index for each row index
   * @param page_row_offsets Host vector row offsets of each page
   * @param dtype The data type of the column
   * @param stream CUDA stream
   * @param mr Device memory resource
   *
   * @return A pair containing the output data buffer and nullmask
   */
  template <typename T>
  [[nodiscard]] std::pair<rmm::device_buffer, rmm::device_buffer> build_data_and_nullmask(
    mutable_column_view input_column,
    bitmask_type const* page_nullmask,
    cudf::device_span<size_type const> page_indices,
    cudf::host_span<size_type const> page_row_offsets,
    cudf::data_type dtype,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
    requires(not cudf::is_compound<T>())
  {
    // Total number of pages in the column
    size_type const total_pages = page_row_offsets.size() - 1;

    // Buffer for output data
    auto output_data = rmm::device_buffer(cudf::size_of(dtype) * total_rows, stream, mr);

    // For each row index, copy over the min/max page stat value from the corresponding page.
    thrust::gather(rmm::exec_policy_nosync(stream),
                   page_indices.begin(),
                   page_indices.end(),
                   input_column.template begin<T>(),
                   reinterpret_cast<T*>(output_data.data()));

    // Buffer for output bitmask. Set all bits valid
    auto output_nullmask = cudf::create_null_mask(total_rows, mask_state::ALL_VALID, stream, mr);

    // For each input page, invalidate the null mask for corresponding rows if needed.
    std::for_each(thrust::counting_iterator(0),
                  thrust::counting_iterator(total_pages),
                  [&](auto const page_idx) {
                    if (not bit_is_set(page_nullmask, page_idx)) {
                      cudf::set_null_mask(static_cast<bitmask_type*>(output_nullmask.data()),
                                          page_row_offsets[page_idx],
                                          page_row_offsets[page_idx + 1],
                                          false,
                                          stream);
                    }
                  });

    return {std::move(output_data), std::move(output_nullmask)};
  }

  /**
   * @brief Transforms a page-level stats column to a row-level stats column for string type
   *
   * @param host_strings Host span of cudf::string_view values in the input page-level host column
   * @param host_chars Host span of string data of the input page-level host column
   * @param host_nullmask Nullmask of the input page-level host column
   * @param page_indices Device vector containing the page index for each row index
   * @param page_row_offsets Host vector row offsets of each page
   * @param stream CUDA stream
   * @param mr Device memory resource
   *
   * @return A pair containing the output data buffer and nullmask
   */
  [[nodiscard]] std::
    tuple<rmm::device_buffer, rmm::device_uvector<cudf::size_type>, rmm::device_buffer>
    build_string_data_and_nullmask(cudf::host_span<cudf::string_view const> host_strings,
                                   cudf::host_span<char const> host_chars,
                                   bitmask_type const* host_page_nullmask,
                                   cudf::device_span<size_type const> page_indices,
                                   cudf::host_span<size_type const> page_row_offsets,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr) const
  {
    // Total number of pages in the column
    size_type const total_pages = page_row_offsets.size() - 1;

    // Construct device vectors containing page-level (input) string data, and offsets and sizes
    auto [page_str_chars, page_str_offsets, page_str_sizes] =
      host_column<cudf::string_view>::make_strings_children(host_strings, host_chars, stream, mr);

    // Buffer for row-level string sizes (output).
    auto row_str_sizes = rmm::device_uvector<size_t>(total_rows, stream, mr);
    // Gather string sizes from page to row level
    thrust::gather(rmm::exec_policy_nosync(stream),
                   page_indices.begin(),
                   page_indices.end(),
                   page_str_sizes.begin(),
                   row_str_sizes.begin());

    // Total bytes in the output chars buffer
    auto const total_bytes = thrust::reduce(rmm::exec_policy(stream),
                                            row_str_sizes.begin(),
                                            row_str_sizes.end(),
                                            size_t{0},
                                            cuda::std::plus<size_t>());

    CUDF_EXPECTS(
      total_bytes <= cuda::std::numeric_limits<cudf::size_type>::max(),
      "The strings child of the page statistics column cannot exceed the column size limit");

    // page-level strings nullmask (input)
    auto const input_nullmask = host_page_nullmask;

    // Buffer for row-level strings nullmask (output). Initialize to all bits set.
    auto output_nullmask = cudf::create_null_mask(total_rows, mask_state::ALL_VALID, stream, mr);

    // For each input page, invalidate the null mask for corresponding rows if needed.
    std::for_each(thrust::counting_iterator(0),
                  thrust::counting_iterator(total_pages),
                  [&](auto const page_idx) {
                    if (not bit_is_set(input_nullmask, page_idx)) {
                      cudf::set_null_mask(static_cast<bitmask_type*>(output_nullmask.data()),
                                          page_row_offsets[page_idx],
                                          page_row_offsets[page_idx + 1],
                                          false,
                                          stream);
                    }
                  });

    // Buffer for row-level string offsets (output).
    auto row_str_offsets =
      cudf::detail::make_zeroed_device_uvector_async<cudf::size_type>(total_rows + 1, stream, mr);
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                           row_str_sizes.begin(),
                           row_str_sizes.end(),
                           row_str_offsets.begin() + 1);

    // Buffer for row-level string chars (output).
    auto row_str_chars = rmm::device_buffer(total_bytes, stream, mr);

    // Iterator for input (page-level) string chars
    auto src_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_t>(0),
      cuda::proclaim_return_type<char*>(
        [chars        = page_str_chars.begin(),
         offsets      = page_str_offsets.begin(),
         page_indices = page_indices.begin()] __device__(size_t index) {
          auto const page_index = page_indices[index];
          return chars + offsets[page_index];
        }));

    // Iterator for output (row-level) string chars
    auto dst_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_t>(0),
      cuda::proclaim_return_type<char*>(
        [chars   = reinterpret_cast<char*>(row_str_chars.data()),
         offsets = row_str_offsets.begin()] __device__(size_t index) {
          return chars + offsets[index];
        }));

    // Iterator for string sizes
    auto size_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_t>(0),
      cuda::proclaim_return_type<size_t>(
        [sizes = row_str_sizes.begin()] __device__(size_t index) { return sizes[index]; }));

    // Gather page-level string chars to row-level string chars
    cudf::detail::batched_memcpy_async(src_iter, dst_iter, size_iter, total_rows, stream);

    // Return row-level (output) strings children and the nullmask
    return std::tuple{
      std::move(row_str_chars), std::move(row_str_offsets), std::move(output_nullmask)};
  }

  /**
   * @brief Builds two device columns storing the corresponding page-level statistics (min, max)
   *        respectively of a column at each row index.
   *
   * @tparam T underlying type of the column
   * @param schema_idx Column schema index
   * @param dtype Column data type
   * @param stream CUDA stream
   * @param mr Device memory resource
   *
   * @return A pair of device columns with min and max value from page statistics for each row
   */
  template <typename T>
  std::pair<std::unique_ptr<column>, std::unique_ptr<column>> operator()(
    cudf::size_type schema_idx,
    cudf::data_type dtype,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    // List, Struct, Dictionary types are not supported
    if constexpr (cudf::is_compound<T>() and not cuda::std::is_same_v<T, string_view>) {
      CUDF_FAIL("Compound types other than strings do not have statistics");
    } else {
      // Compute column chunk level page count offsets, and page level row counts and row offsets.
      auto const [page_row_counts, page_row_offsets, col_chunk_page_offsets] =
        compute_page_row_counts_and_offsets(
          per_file_metadata, row_group_indices, schema_idx, stream);

      CUDF_EXPECTS(
        page_row_offsets.back() == total_rows,
        "The number of rows must be equal across row groups and pages within row groups");

      auto const total_pages = col_chunk_page_offsets.back();

      // Create host columns with page-level min, max values
      host_column<T> min(total_pages, stream);
      host_column<T> max(total_pages, stream);

      // Populate the host columns with page-level min, max statistics from the page index
      auto page_offset_idx = 0;
      // For all row data sources
      std::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator(row_group_indices.size()),
        [&](auto src_idx) {
          // For all column chunks in this source
          auto const& rg_indices = row_group_indices[src_idx];
          std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto rg_idx) {
            auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
            // Find colchunk_iter in row_group.columns. Guaranteed to be found as already verified
            // in compute_page_row_counts_and_offsets()
            auto colchunk_iter = std::find_if(
              row_group.columns.begin(),
              row_group.columns.end(),
              [schema_idx](ColumnChunk const& col) { return col.schema_idx == schema_idx; });

            auto const& colchunk               = *colchunk_iter;
            auto const& column_index           = colchunk.column_index.value();
            auto const num_pages_in_colchunk   = column_index.min_values.size();
            auto const page_offset_in_colchunk = col_chunk_page_offsets[page_offset_idx++];

            // For all pages in this column chunk
            std::for_each(
              thrust::counting_iterator<size_t>(0),
              thrust::counting_iterator(num_pages_in_colchunk),
              [&](auto page_idx) {
                auto const& min_value = column_index.min_values[page_idx];
                auto const& max_value = column_index.max_values[page_idx];
                // Translate binary data to Type then to <T>
                min.set_index(
                  page_offset_in_colchunk + page_idx, min_value, colchunk.meta_data.type);
                max.set_index(
                  page_offset_in_colchunk + page_idx, max_value, colchunk.meta_data.type);
              });
          });
        });

      // Construct a row indices mapping based on page row counts and offsets
      auto const page_indices =
        make_page_indices_async(page_row_counts, page_row_offsets, total_rows, stream);

      // For non-strings columns, directly gather the page-level column data and bitmask to the
      // row-level.
      if constexpr (not cuda::std::is_same_v<T, cudf::string_view>) {
        // Move host columns to device
        auto mincol = min.to_device(dtype, stream, mr);
        auto maxcol = max.to_device(dtype, stream, mr);

        // Convert page-level min and max columns to row-level min and max columns by gathering
        // values based on page-level row offsets
        auto [min_data, min_bitmask] = build_data_and_nullmask<T>(mincol->mutable_view(),
                                                                  min.null_mask.data(),
                                                                  page_indices,
                                                                  page_row_offsets,
                                                                  dtype,
                                                                  stream,
                                                                  mr);
        auto [max_data, max_bitmask] = build_data_and_nullmask<T>(maxcol->mutable_view(),
                                                                  max.null_mask.data(),
                                                                  page_indices,
                                                                  page_row_offsets,
                                                                  dtype,
                                                                  stream,
                                                                  mr);

        // Count nulls in min and max columns
        auto const min_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(min_bitmask.data()), 0, total_rows, stream);
        auto const max_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(max_bitmask.data()), 0, total_rows, stream);

        // Return min and max device columns
        return {std::make_unique<column>(
                  dtype, total_rows, std::move(min_data), std::move(min_bitmask), min_nulls),
                std::make_unique<column>(
                  dtype, total_rows, std::move(max_data), std::move(max_bitmask), max_nulls)};
      }
      // For strings columns, gather the page-level string offsets and bitmask to row-level
      // directly and gather string chars using a batched memcpy.
      else {
        auto [min_data, min_offsets, min_nullmask] = build_string_data_and_nullmask(
          min.val, min.chars, min.null_mask.data(), page_indices, page_row_offsets, stream, mr);
        auto [max_data, max_offsets, max_nullmask] = build_string_data_and_nullmask(
          max.val, max.chars, max.null_mask.data(), page_indices, page_row_offsets, stream, mr);

        // Count nulls in min and max columns
        auto const min_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(min_nullmask.data()), 0, total_rows, stream);
        auto const max_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(max_nullmask.data()), 0, total_rows, stream);

        // Return min and max device strings columns
        return {
          cudf::make_strings_column(
            total_rows,
            std::make_unique<column>(std::move(min_offsets), rmm::device_buffer{0, stream, mr}, 0),
            std::move(min_data),
            min_nulls,
            std::move(min_nullmask)),
          cudf::make_strings_column(
            total_rows,
            std::make_unique<column>(std::move(max_offsets), rmm::device_buffer{0, stream, mr}, 0),
            std::move(max_data),
            max_nulls,
            std::move(max_nullmask))};
      }
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> aggregate_reader_metadata::build_row_mask_with_page_index_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<cudf::data_type const> output_dtypes,
  cudf::host_span<cudf::size_type const> output_column_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Return if empty row group indices
  if (row_group_indices.empty()) { return cudf::make_empty_column(cudf::type_id::BOOL8); }

  // Check if we have page index for all columns in all row groups
  auto const has_page_index =
    compute_has_page_index(per_file_metadata, row_group_indices, output_column_schemas);

  // Return if page index is not present
  CUDF_EXPECTS(has_page_index,
               "Page pruning requires the Parquet page index for all output columns",
               std::runtime_error);

  // Total number of rows
  auto const total_rows = std::accumulate(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(row_group_indices.size()),
    size_t{0},
    [&](auto sum, auto const src_index) {
      auto const& rg_indices = row_group_indices[src_index];
      return std::accumulate(
        rg_indices.begin(), rg_indices.end(), sum, [&](auto subsum, auto const rg_index) {
          CUDF_EXPECTS(subsum + per_file_metadata[src_index].row_groups[rg_index].num_rows <=
                         std::numeric_limits<size_type>::max(),
                       "Total rows exceed the maximum value");
          return subsum + per_file_metadata[src_index].row_groups[rg_index].num_rows;
        });
    });

  auto const num_columns = output_dtypes.size();

  // Get a boolean mask indicating which columns will participate in stats based filtering
  auto const stats_columns_mask =
    parquet::detail::stats_columns_collector{filter.get(),
                                             static_cast<size_type>(output_dtypes.size())}
      .get_stats_columns_mask();

  // Return early if no columns will participate in stats based page filtering
  if (stats_columns_mask.empty()) {
    auto const scalar_true = cudf::numeric_scalar<bool>(true, true, stream);
    return cudf::make_column_from_scalar(scalar_true, total_rows, stream, mr);
  }

  // Convert page statistics to a table
  // where min(col[i]) = columns[i*2], max(col[i])=columns[i*2+1]
  // For each column, it contains total number of rows from all row groups.
  page_stats_caster const stats_col{
    static_cast<size_type>(total_rows), per_file_metadata, row_group_indices};

  std::vector<std::unique_ptr<column>> columns;
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(num_columns),
    [&](auto col_idx) {
      auto const schema_idx = output_column_schemas[col_idx];
      auto const& dtype     = output_dtypes[col_idx];
      // Only participating columns and comparable types except fixed point are supported
      if (not stats_columns_mask[col_idx] or
          (cudf::is_compound(dtype) && dtype.id() != cudf::type_id::STRING)) {
        // Placeholder for unsupported types and non-participating columns
        columns.push_back(cudf::make_numeric_column(
          data_type{cudf::type_id::BOOL8}, total_rows, rmm::device_buffer{}, 0, stream, mr));
        columns.push_back(cudf::make_numeric_column(
          data_type{cudf::type_id::BOOL8}, total_rows, rmm::device_buffer{}, 0, stream, mr));
        return;
      }
      auto [min_col, max_col] = cudf::type_dispatcher<dispatch_storage_type>(
        dtype, stats_col, schema_idx, dtype, stream, mr);
      columns.push_back(std::move(min_col));
      columns.push_back(std::move(max_col));
    });

  auto stats_table = cudf::table(std::move(columns));

  // Converts AST to StatsAST with reference to min, max columns in above `stats_table`.
  parquet::detail::stats_expression_converter const stats_expr{
    filter.get(), static_cast<size_type>(output_dtypes.size()), stream};

  // Filter the input table using AST expression and return the (BOOL8) predicate column.
  return cudf::detail::compute_column(stats_table, stats_expr.get_stats_expr().get(), stream, mr);
}

template <typename ColumnView>
std::vector<bool> aggregate_reader_metadata::compute_data_page_mask(
  ColumnView const& row_mask,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<input_column_info const> input_columns,
  cudf::size_type row_mask_offset,
  rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(row_mask.type().id() == cudf::type_id::BOOL8,
               "Input row bitmask should be of type BOOL8");

  auto const num_columns = input_columns.size();

  // Collect column schema indices from the input columns.
  auto column_schema_indices = std::vector<size_type>(input_columns.size());
  std::transform(
    input_columns.begin(), input_columns.end(), column_schema_indices.begin(), [](auto const& col) {
      return col.schema_idx;
    });
  auto const has_page_index =
    compute_has_page_index(per_file_metadata, row_group_indices, column_schema_indices);

  // Return early if page index is not present
  if (not has_page_index) {
    CUDF_LOG_WARN("Encountered missing Parquet page index for one or more output columns");
    return {};  // An empty data page mask indicates all pages are required
  }

  // Compute page row offsets and column chunk page offsets for each column
  std::vector<size_type> page_row_offsets;
  std::vector<size_type> col_page_offsets;
  col_page_offsets.reserve(num_columns + 1);
  col_page_offsets.push_back(0);

  size_type max_page_size = 0;

  if (num_columns == 1) {
    auto const schema_idx   = column_schema_indices.front();
    size_type col_num_pages = 0;
    std::tie(page_row_offsets, col_num_pages, max_page_size) =
      compute_page_row_offsets(per_file_metadata, row_group_indices, schema_idx);
    // Add 1 to include the the 0th page's offset for each column
    col_page_offsets.emplace_back(col_num_pages + 1);
  } else {
    std::vector<std::future<std::tuple<std::vector<size_type>, size_type, size_type>>>
      page_row_offsets_tasks;
    page_row_offsets_tasks.reserve(num_columns);

    std::for_each(thrust::counting_iterator<size_t>(0),
                  thrust::counting_iterator(num_columns),
                  [&](auto const col_idx) {
                    page_row_offsets_tasks.emplace_back(
                      cudf::detail::host_worker_pool().submit_task([&, col_idx = col_idx] {
                        return compute_page_row_offsets(
                          per_file_metadata, row_group_indices, column_schema_indices[col_idx]);
                      }));
                  });

    // Collect results from all tasks
    std::for_each(page_row_offsets_tasks.begin(), page_row_offsets_tasks.end(), [&](auto& task) {
      auto [col_page_row_offsets, col_num_pages, col_max_page_size] = std::move(task).get();
      page_row_offsets.insert(page_row_offsets.end(),
                              std::make_move_iterator(col_page_row_offsets.begin()),
                              std::make_move_iterator(col_page_row_offsets.end()));
      max_page_size = std::max<size_type>(max_page_size, col_max_page_size);
      // Add 1 to include the the 0th page's offset for each column
      col_page_offsets.emplace_back(col_page_offsets.back() + col_num_pages + 1);
    });
  }

  auto const total_rows  = page_row_offsets.back();
  auto const total_pages = page_row_offsets.size() - num_columns;

  CUDF_EXPECTS(row_mask_offset + total_rows <= row_mask.size(),
               "Mismatch in total rows in input row mask and row groups",
               std::invalid_argument);

  // Return an empty vector if all rows are invalid or all rows are required
  if (row_mask.null_count(row_mask_offset, row_mask_offset + total_rows, stream) == total_rows or
      thrust::all_of(rmm::exec_policy(stream),
                     row_mask.template begin<bool>() + row_mask_offset,
                     row_mask.template begin<bool>() + row_mask_offset + total_rows,
                     cuda::std::identity{})) {
    return {};
  }

  if constexpr (cuda::std::is_same_v<ColumnView, cudf::mutable_column_view>) {
    if (row_mask.nullable()) {
      thrust::for_each(rmm::exec_policy_nosync(stream),
                       thrust::counting_iterator(row_mask_offset),
                       thrust::counting_iterator(row_mask_offset + total_rows),
                       [row_mask  = row_mask.template begin<bool>(),
                        null_mask = row_mask.null_mask()] __device__(auto const row_idx) {
                         if (not bit_is_set(null_mask, row_idx)) { row_mask[row_idx] = true; }
                       });
    }
  } else {
    CUDF_EXPECTS(not row_mask.nullable() or row_mask.null_count() == 0,
                 "Row mask must not contain nulls for payload columns");
  }

  // Total number of surviving pages across all columns
  std::atomic<size_t> total_surviving_pages{0};

  // Tasks to compute data page mask for each column
  std::vector<std::future<std::vector<bool>>> data_page_mask_tasks;
  data_page_mask_tasks.reserve(num_columns);

  // Host row mask data
  auto const is_row_required = cudf::detail::make_host_vector(
    device_span<uint8_t const>(row_mask.template data<uint8_t>() + row_mask_offset, total_rows),
    stream);

  // For all columns, look up which pages contain at least one required row. i.e.
  // !validity_it[row_idx] or is_row_required[row_idx] satisfies, and add its byte range to the
  // output list of byte ranges for the column.
  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(num_columns),
                [&](auto const col_idx) {
                  data_page_mask_tasks.emplace_back(
                    cudf::detail::host_worker_pool().submit_task([&, col_idx = col_idx] {
                      // Construct a row indices mapping based on page row counts and offsets
                      auto const total_pages_this_column =
                        col_page_offsets[col_idx + 1] - col_page_offsets[col_idx] - 1;
                      auto col_data_page_mask = std::vector<bool>(total_pages_this_column, false);
                      // Number of final filtered pages for this column
                      size_t num_surviving_pages_this_column = 0;
                      // For all rows
                      for (auto row_idx = 0; row_idx < total_rows; ++row_idx) {
                        // If this row is required, add its page index to the output list.
                        if (is_row_required[row_idx]) {
                          // binary search to find the page index this row_idx belongs to and set
                          // the page index to true page_indices
                          auto const offsets = cudf::host_span<size_type const>(
                            page_row_offsets.data() + col_page_offsets[col_idx],
                            col_page_offsets[col_idx + 1] - col_page_offsets[col_idx]);
                          auto const page_itr =
                            std::upper_bound(offsets.begin(), offsets.end(), row_idx);
                          CUDF_EXPECTS(page_itr != offsets.begin() and page_itr != offsets.end(),
                                       "Invalid page index");
                          auto const page_idx = std::distance(offsets.begin(), page_itr) - 1;
                          col_data_page_mask[page_idx] = true;
                          num_surviving_pages_this_column++;
                          // Move row_idx to the last row of this page
                          row_idx = offsets[page_idx + 1] - 1;
                        }
                      }
                      total_surviving_pages.fetch_add(num_surviving_pages_this_column);
                      return col_data_page_mask;
                    }));
                });

  // Vector to hold data page mask for each column
  auto data_page_mask = std::vector<bool>();
  data_page_mask.reserve(total_pages);

  std::for_each(data_page_mask_tasks.begin(), data_page_mask_tasks.end(), [&](auto& task) {
    auto col_data_page_mask = std::move(task).get();
    data_page_mask.insert(data_page_mask.end(),
                          std::make_move_iterator(col_data_page_mask.begin()),
                          std::make_move_iterator(col_data_page_mask.end()));
  });

  CUDF_EXPECTS(
    total_surviving_pages <= total_pages,
    "Number of surviving pages must be less than or equal to the total number of input pages");

  return data_page_mask;
}

// Instantiate the templates with ColumnView as cudf::column_view and cudf::mutable_column_view
template std::vector<bool> aggregate_reader_metadata::compute_data_page_mask<cudf::column_view>(
  cudf::column_view const& row_mask,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<input_column_info const> input_columns,
  cudf::size_type row_mask_offset,
  rmm::cuda_stream_view stream) const;

template std::vector<bool> aggregate_reader_metadata::compute_data_page_mask<
  cudf::mutable_column_view>(cudf::mutable_column_view const& row_mask,
                             cudf::host_span<std::vector<size_type> const> row_group_indices,
                             cudf::host_span<input_column_info const> input_columns,
                             cudf::size_type row_mask_offset,
                             rmm::cuda_stream_view stream) const;

}  // namespace cudf::io::parquet::experimental::detail
