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
#include "io/parquet/reader_impl_helpers.hpp"
#include "io/parquet/stats_filter_helpers.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/types.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/unique.h>

#include <algorithm>
#include <limits>
#include <numeric>

namespace cudf::io::parquet::experimental::detail {

using metadata_base     = parquet::detail::metadata;
using stats_caster_base = parquet::detail::stats_caster_base;
using string_index_pair = parquet::detail::string_index_pair;

namespace {

/**
 * @brief Make a device vector where each row contains the index of the page it belongs to
 */
[[nodiscard]] rmm::device_uvector<size_type> make_page_indices(
  cudf::host_span<cudf::size_type const> page_row_counts,
  cudf::host_span<cudf::size_type const> page_row_offsets,
  cudf::size_type total_rows,
  rmm::cuda_stream_view stream)
{
  auto mr = cudf::get_current_device_resource_ref();

  // Move page-level row counts and offsets to device
  auto row_counts  = cudf::detail::make_device_uvector_async(page_row_counts, stream, mr);
  auto row_offsets = cudf::detail::make_device_uvector_async(page_row_offsets, stream, mr);

  // Generate row index mapping
  auto page_indices =
    cudf::detail::make_zeroed_device_uvector_async<cudf::size_type>(total_rows, stream, mr);
  thrust::scatter_if(rmm::exec_policy_nosync(stream),
                     thrust::counting_iterator<size_type>(0),
                     thrust::counting_iterator<size_type>(row_counts.size()),
                     row_offsets.begin(),
                     row_counts.begin(),
                     page_indices.begin());

  // Fill gaps with previous values
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                         page_indices.begin(),
                         page_indices.end(),
                         page_indices.begin(),
                         thrust::maximum<cudf::size_type>());
  return page_indices;
}

/**
 * @brief Compute page row counts and page row offsets and colum chunk page (count) offsets for a
 * given column schema index
 */
[[nodiscard]] auto make_page_row_counts_and_offsets(
  cudf::host_span<metadata_base const> per_file_metadata,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  size_type schema_idx)
{
  std::vector<size_type> page_row_counts;
  std::vector<size_type> page_row_offsets{0};
  std::vector<size_type> col_chunk_page_offsets{0};

  // For all sources
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(row_group_indices.size()),
    [&](auto src_idx) {
      auto const& rg_indices = row_group_indices[src_idx];
      // For all row groups in this source
      std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto rg_idx) {
        auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
        auto col              = std::find_if(
          row_group.columns.begin(), row_group.columns.end(), [schema_idx](ColumnChunk const& col) {
            return col.schema_idx == schema_idx;
          });

        if (col != std::end(row_group.columns) and col->offset_index.has_value()) {
          CUDF_EXPECTS(col->column_index.has_value(),
                       "Both offset and column indexes must be present");

          auto const& offset_index       = col->offset_index.value();
          auto const& column_index       = col->column_index.value();
          auto const row_group_num_pages = offset_index.page_locations.size();

          CUDF_EXPECTS(column_index.min_values.size() == column_index.max_values.size(),
                       "page min and max values should be of same size");
          CUDF_EXPECTS(column_index.min_values.size() == offset_index.page_locations.size(),
                       "mismatch between size of min/max page values and the size of page "
                       "locations");
          // Get page offsets for this row group
          col_chunk_page_offsets.emplace_back(col_chunk_page_offsets.back() +
                                              offset_index.page_locations.size());

          // For all pages in this row group, Get row counts and offsets.
          std::for_each(
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator(row_group_num_pages),
            [&](auto const page_idx) {
              int64_t const first_row_idx = offset_index.page_locations[page_idx].first_row_index;
              int64_t const last_row_idx =
                (page_idx < row_group_num_pages - 1)
                  ? offset_index.page_locations[page_idx + 1].first_row_index
                  : row_group.num_rows;

              page_row_counts.emplace_back(last_row_idx - first_row_idx);
              page_row_offsets.emplace_back(page_row_offsets.back() + page_row_counts.back());
            });
        }
      });
    });

  CUDF_EXPECTS(col_chunk_page_offsets.back() > 0,
               "Page index is not present for page pruning",
               std::runtime_error);

  return std::tuple{
    std::move(page_row_counts), std::move(page_row_offsets), std::move(col_chunk_page_offsets)};
}

/**
 * @brief Construct a vector of all required data pages from the page row counts
 */
[[nodiscard]] auto all_required_data_pages(
  cudf::host_span<std::vector<size_type> const> page_row_counts)
{
  std::vector<thrust::host_vector<bool>> all_required_data_pages;
  all_required_data_pages.reserve(page_row_counts.size());
  std::for_each(page_row_counts.begin(), page_row_counts.end(), [&](auto const& col_page_counts) {
    all_required_data_pages.emplace_back(col_page_counts.size(), true);
  });

  return all_required_data_pages;
};

/**
 * @brief Converts page-level statistics to 2 device columns - min, max values. Each column has
 *        number of rows equal to the total rows in all row groups.
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

  // Creates device columns from column statistics (min, max)
  template <typename T>
  std::pair<std::unique_ptr<column>, std::unique_ptr<column>> operator()(
    int schema_idx,
    cudf::data_type dtype,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    // List, Struct, Dictionary types are not supported
    if constexpr (cudf::is_compound<T>() && !std::is_same_v<T, string_view>) {
      CUDF_FAIL("Compound types do not have statistics");
    } else {
      // Compute column chunk level page count offsets, and page level row counts and row offsets.
      auto const [page_row_counts, page_row_offsets, col_chunk_page_offsets] =
        make_page_row_counts_and_offsets(per_file_metadata, row_group_indices, schema_idx);

      CUDF_EXPECTS(page_row_offsets.back() == total_rows, "Mismatch in total rows");

      // Create host columns with page-level min, max values
      auto const total_pages = col_chunk_page_offsets.back();
      host_column<T> min(total_pages, stream);
      host_column<T> max(total_pages, stream);

      auto page_offset_idx = 0;

      // For each source
      std::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator(row_group_indices.size()),
        [&, col_chunk_page_offsets = col_chunk_page_offsets](auto src_idx) {
          // For all row groups in this source
          auto const& rg_indices = row_group_indices[src_idx];
          std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto rg_idx) {
            auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
            auto col              = std::find_if(
              row_group.columns.begin(),
              row_group.columns.end(),
              [schema_idx](ColumnChunk const& col) { return col.schema_idx == schema_idx; });

            // No need to check column_index and offset_index again
            if (col != std::end(row_group.columns)) {
              auto const& colchunk               = *col;
              auto const& column_index           = colchunk.column_index.value();
              auto const& offset_index           = colchunk.offset_index.value();
              auto const num_pages_in_colchunk   = column_index.min_values.size();
              auto const page_offset_in_colchunk = col_chunk_page_offsets[page_offset_idx++];

              // For all pages in this chunk
              std::for_each(
                thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(num_pages_in_colchunk),
                [&](auto page_idx) {
                  // To support deprecated min, max fields.
                  auto const& min_value = column_index.min_values[page_idx];
                  auto const& max_value = column_index.min_values[page_idx];
                  // Translate binary data to Type then to <T>
                  min.set_index(
                    page_offset_in_colchunk + page_idx, min_value, colchunk.meta_data.type);
                  max.set_index(
                    page_offset_in_colchunk + page_idx, max_value, colchunk.meta_data.type);
                });
            }
          });
        });

      // Construct a row indices mapping based on page row counts and offsets
      auto const page_indices =
        make_page_indices(page_row_counts, page_row_offsets, total_rows, stream);

      // For non-strings columns, directly gather the page-level column data and bitmask to the
      // row-level.
      if constexpr (not std::is_same_v<T, cudf::string_view>) {
        // Lambda function to build a row-level device column from a page-level column
        auto const build_data_and_nullmask = [&, page_row_offsets = page_row_offsets](
                                               mutable_column_view column,
                                               bitmask_type const* page_nullmask,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr) {
          // Buffer for output data
          auto output_data = rmm::device_buffer(cudf::size_of(dtype) * total_rows, stream, mr);

          // For each row index, copy over the min/max page stat value from the corresponding page.
          thrust::gather(rmm::exec_policy_nosync(stream),
                         page_indices.begin(),
                         page_indices.end(),
                         column.template begin<T>(),
                         reinterpret_cast<T*>(output_data.data()));

          // Buffer for output bitmask. Set all bits valid
          auto output_nullmask =
            cudf::create_null_mask(total_rows, mask_state::ALL_VALID, stream, mr);

          // For each input page, invalidate the null mask for corresponding rows if needed.
          std::for_each(thrust::counting_iterator(0),
                        thrust::counting_iterator(total_pages),
                        [&, page_row_offsets = page_row_offsets.data()](auto const page_idx) {
                          if (not bit_is_set(page_nullmask, page_idx)) {
                            cudf::set_null_mask(static_cast<bitmask_type*>(output_nullmask.data()),
                                                page_row_offsets[page_idx],
                                                page_row_offsets[page_idx + 1],
                                                false,
                                                stream);
                          }
                        });

          return std::pair{std::move(output_data), std::move(output_nullmask)};
        };

        // Move host columns to device
        auto mincol = min.to_device(dtype, stream, mr);
        auto maxcol = max.to_device(dtype, stream, mr);

        // Convert page-level min and max columns to row-level min and max columns by gathering
        // values based on page-level row offsets
        auto [min_data, min_bitmask] =
          build_data_and_nullmask(mincol->mutable_view(), min.null_mask.data(), stream, mr);
        auto [max_data, max_bitmask] =
          build_data_and_nullmask(maxcol->mutable_view(), min.null_mask.data(), stream, mr);

        // Count nulls in min and max columns
        auto const min_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(min_bitmask.data()), 0, total_rows, stream);
        auto const max_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(max_bitmask.data()), 0, total_rows, stream);

        stream.synchronize();

        // Return min and max device columns
        return {std::make_unique<column>(
                  dtype, total_rows, std::move(min_data), std::move(min_bitmask), min_nulls),
                std::make_unique<column>(
                  dtype, total_rows, std::move(max_data), std::move(max_bitmask), max_nulls)};
      }
      // For strings columns, gather the page-level string offsets and bitmask to row-level
      // directly and gather string chars using a batched memcpy.
      else {
        // Lambda function to build a row-level device strings children and nullmask from the
        // page-level string host column
        auto const build_strings_children_and_nullmask = [&, page_row_offsets = page_row_offsets](
                                                           host_column<T> const& host_col,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr) {
          // Construct device vectors containing page-level (input) string children and sizes.
          auto [page_str_chars, page_str_sizes, page_str_offsets] = [&]() {
            auto const& host_strings    = host_col.val;
            auto const total_char_count = std::accumulate(
              host_strings.begin(), host_strings.end(), size_t{0}, [](auto sum, auto const& str) {
                return sum + str.size_bytes();
              });
            auto chars = cudf::detail::make_empty_host_vector<char>(total_char_count, stream);
            auto sizes = cudf::detail::make_empty_host_vector<cudf::size_type>(total_pages, stream);
            auto offsets =
              cudf::detail::make_empty_host_vector<std::size_t>(total_pages + 1, stream);
            offsets.push_back(0);
            for (auto const& str : host_strings) {
              auto tmp =
                str.empty() ? std::string_view{} : std::string_view(str.data(), str.size_bytes());
              chars.insert(chars.end(), std::cbegin(tmp), std::cend(tmp));
              sizes.push_back(tmp.length());
              offsets.push_back(offsets.back() + tmp.length());
            }
            return std::tuple{cudf::detail::make_device_uvector_async(chars, stream, mr),
                              cudf::detail::make_device_uvector_async(sizes, stream, mr),
                              cudf::detail::make_device_uvector_async(offsets, stream, mr)};
          }();

          // Buffer for row-level string sizes (output).
          auto row_str_sizes = rmm::device_uvector<std::size_t>(total_rows, stream, mr);
          // Gather string sizes from page to row level
          thrust::gather(rmm::exec_policy_nosync(stream),
                         page_indices.begin(),
                         page_indices.end(),
                         page_str_sizes.begin(),
                         row_str_sizes.begin());

          // page-level strings nullmask (input)
          auto const input_nullmask = host_col.null_mask.data();

          // Buffer for row-level strings nullmask (output). Initialize to all bits set.
          auto output_nullmask =
            cudf::create_null_mask(total_rows, mask_state::ALL_VALID, stream, mr);

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
            cudf::detail::make_zeroed_device_uvector_async<std::size_t>(total_rows + 1, stream, mr);
          thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                                 row_str_sizes.begin(),
                                 row_str_sizes.end(),
                                 row_str_offsets.begin() + 1);

          // Total bytes in the output chars buffer
          auto const total_bytes = row_str_offsets.back_element(stream);

          // Buffer for row-level string chars (output).
          auto row_str_chars = rmm::device_buffer(total_bytes, stream, mr);

          // Iterator for input (page-level) string chars
          auto src_iter = thrust::make_transform_iterator(
            thrust::make_counting_iterator<std::size_t>(0),
            cuda::proclaim_return_type<char*>(
              [chars        = page_str_chars.begin(),
               offsets      = page_str_offsets.begin(),
               page_indices = page_indices.begin()] __device__(std::size_t index) {
                auto const page_index = page_indices[index];
                return chars + offsets[page_index];
              }));

          // Iterator for output (row-level) string chars
          auto dst_iter = thrust::make_transform_iterator(
            thrust::make_counting_iterator<std::size_t>(0),
            cuda::proclaim_return_type<char*>(
              [chars   = reinterpret_cast<char*>(row_str_chars.data()),
               offsets = row_str_offsets.begin()] __device__(std::size_t index) {
                return chars + offsets[index];
              }));

          // Iterator for string sizes
          auto size_iter = thrust::make_transform_iterator(
            thrust::make_counting_iterator<std::size_t>(0),
            cuda::proclaim_return_type<size_t>([sizes = row_str_sizes.begin()] __device__(
                                                 std::size_t index) { return sizes[index]; }));

          // Gather page-level string chars to row-level string chars
          cudf::detail::batched_memcpy_async(src_iter, dst_iter, size_iter, total_rows, stream);

          // Return row-level (output) strings children and the nullmask
          return std::tuple{
            std::move(row_str_chars), std::move(row_str_offsets), std::move(output_nullmask)};
        };

        auto [min_data, min_offsets, min_nullmask] =
          build_strings_children_and_nullmask(min, stream, mr);
        auto [max_data, max_offsets, max_nullmask] =
          build_strings_children_and_nullmask(min, stream, mr);

        // Count nulls in min and max columns
        auto const min_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(min_nullmask.data()), 0, total_rows, stream);
        auto const max_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(max_nullmask.data()), 0, total_rows, stream);

        stream.synchronize();

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

std::unique_ptr<cudf::column> aggregate_reader_metadata::filter_data_pages_with_stats(
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
  auto const has_page_index = std::all_of(
    output_column_schemas.begin(), output_column_schemas.end(), [&](auto const schema_idx) {
      return std::all_of(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator(row_group_indices.size()),
        [&](auto const src_index) {
          auto const& rg_indices = row_group_indices[src_index];
          return std::all_of(rg_indices.begin(), rg_indices.end(), [&](auto const& rg_index) {
            auto const& row_group = per_file_metadata[src_index].row_groups[rg_index];
            auto col              = std::find_if(
              row_group.columns.begin(),
              row_group.columns.end(),
              [schema_idx](ColumnChunk const& col) { return col.schema_idx == schema_idx; });
            return col != per_file_metadata[src_index].row_groups[rg_index].columns.end() and
                   col->offset_index.has_value() and col->column_index.has_value();
          });
        });
    });

  // Return if page index is not present
  CUDF_EXPECTS(has_page_index, "Page index is not present for page pruning", std::runtime_error);

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
      // Only comparable types except fixed point are supported.
      if (cudf::is_compound(dtype) && dtype.id() != cudf::type_id::STRING) {
        // placeholder only for unsupported types.
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
    filter.get(), static_cast<size_type>(output_dtypes.size())};

  // Filter the input table using AST expression and return the (BOOL8) predicate column.
  return cudf::detail::compute_column(stats_table, stats_expr.get_stats_expr().get(), stream, mr);
}

std::vector<thrust::host_vector<bool>> aggregate_reader_metadata::compute_data_page_mask(
  cudf::column_view row_mask,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<cudf::data_type const> output_dtypes,
  cudf::host_span<cudf::size_type const> output_column_schemas,
  rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(row_mask.type().id() == cudf::type_id::BOOL8,
               "Input row bitmask should be of type BOOL8");

  auto const total_rows  = row_mask.size();
  auto const num_columns = output_dtypes.size();

  // Compute page row counts, offsets, and column chunk page offsets for each column
  std::vector<std::vector<size_type>> page_row_counts(num_columns);
  std::vector<std::vector<size_type>> page_row_offsets(num_columns);
  std::vector<std::vector<size_type>> col_chunk_page_offsets(num_columns);
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(num_columns),
    [&](auto col_idx) {
      auto const schema_idx = output_column_schemas[col_idx];
      std::tie(
        page_row_counts[col_idx], page_row_offsets[col_idx], col_chunk_page_offsets[col_idx]) =
        make_page_row_counts_and_offsets(per_file_metadata, row_group_indices, schema_idx);
    });

  CUDF_EXPECTS(page_row_offsets.back().back() == total_rows,
               "Mismatch in total rows in input row mask and row groups",
               std::invalid_argument);

  // Return if all rows are required or all are invalid.
  if (row_mask.null_count() == row_mask.size() or thrust::all_of(rmm::exec_policy(stream),
                                                                 row_mask.begin<bool>(),
                                                                 row_mask.end<bool>(),
                                                                 cuda::std::identity{})) {
    return all_required_data_pages(page_row_counts);
  }

  // Compute total number of input pages
  auto const total_pages =
    std::accumulate(page_row_offsets.cbegin(),
                    page_row_offsets.cend(),
                    size_t{0},
                    [](auto sum, auto const& offsets) { return sum + offsets.size() - 1; });

  auto const mr = cudf::get_current_device_resource_ref();

  // Vector to hold data page mask for each column
  auto data_page_mask = std::vector<thrust::host_vector<bool>>();
  data_page_mask.reserve(num_columns);
  auto total_filtered_pages = size_t{0};

  // For all columns, look up which pages contain at least one required row. i.e.
  // !validity_it[row_idx] or is_row_required[row_idx] satisfies, and add its byte range to the
  // output list of byte ranges for the column.
  for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
    // Construct a row indices mapping based on page row counts and offsets
    auto const total_pages_in_this_column = page_row_counts[col_idx].size();

    auto const page_indices =
      make_page_indices(page_row_counts[col_idx], page_row_offsets[col_idx], total_rows, stream);

    // Device vector to hold page indices with at least one valid row
    rmm::device_uvector<size_type> select_page_indices(total_rows, stream, mr);

    // Copy page indices with at least one valid row
    auto const filtered_pages_end_iter =
      thrust::copy_if(rmm::exec_policy_nosync(stream),
                      page_indices.begin(),
                      page_indices.end(),
                      thrust::counting_iterator<size_type>(0),
                      select_page_indices.begin(),
                      [is_nullable     = row_mask.nullable(),
                       bitmask         = row_mask.null_mask(),
                       is_row_required = row_mask.data<bool>()] __device__(size_type row_index) {
                        auto const is_invalid = is_nullable and not bit_is_set(bitmask, row_index);
                        return is_invalid or is_row_required[row_index];
                      });

    // Remove duplicate page indices across (presorted) rows
    auto const filtered_uniq_page_end_iter = thrust::unique(
      rmm::exec_policy_nosync(stream), select_page_indices.begin(), filtered_pages_end_iter);

    // Number of final filtered pages for this column
    size_t const num_filtered_pages =
      thrust::distance(select_page_indices.begin(), filtered_uniq_page_end_iter);

    total_filtered_pages += num_filtered_pages;

    // Copy the filtered page indices for this column to host
    auto host_select_page_indices = cudf::detail::make_host_vector(
      cudf::device_span<cudf::size_type const>{select_page_indices.data(), num_filtered_pages},
      stream);

    // Vector to data page mask the this column
    auto valid_pages = thrust::host_vector<bool>(total_pages_in_this_column, false);
    std::for_each(host_select_page_indices.begin(),
                  host_select_page_indices.end(),
                  [&](auto const page_idx) { valid_pages[page_idx] = true; });

    data_page_mask.emplace_back(std::move(valid_pages));
  }

  CUDF_EXPECTS(
    total_filtered_pages <= total_pages,
    "Number of filtered pages must be less than or equal to the total number of input pages");

  return data_page_mask;
}

}  // namespace cudf::io::parquet::experimental::detail
