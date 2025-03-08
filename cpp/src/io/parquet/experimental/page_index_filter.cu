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

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/types.hpp>
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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_set>

namespace cudf::experimental::io::parquet::detail {

namespace {

using Type = cudf::io::parquet::detail::Type;

/**
 * @brief Converts page-level statistics to 2 device columns - min, max values. Each column has
 *        number of rows equal to the total rows in all row groups.
 *
 */
struct page_stats_caster : public cudf::io::parquet::detail::stats_caster_base {
  size_type total_rows;
  std::vector<cudf::io::parquet::detail::metadata> const& per_file_metadata;
  host_span<std::vector<size_type> const> row_group_indices;

  page_stats_caster(size_type total_rows,
                    std::vector<cudf::io::parquet::detail::metadata> const& per_file_metadata,
                    host_span<std::vector<size_type> const> row_group_indices)
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
      // Compute row group level page count offsets, and page level row counts and row offsets.
      auto const [row_group_page_offsets, page_row_counts, page_row_offsets] = [&]() {
        std::vector<size_type> row_group_page_offsets{0};
        std::vector<size_type> page_row_counts;
        std::vector<size_type> page_row_offsets{0};

        // For all sources
        std::for_each(
          thrust::counting_iterator<size_t>(0),
          thrust::counting_iterator(row_group_indices.size()),
          [&](auto src_idx) {
            auto const& rg_indices = row_group_indices[src_idx];
            // For all row groups in this source
            std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto rg_idx) {
              auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
              auto col =
                std::find_if(row_group.columns.begin(),
                             row_group.columns.end(),
                             [schema_idx](cudf::io::parquet::detail::ColumnChunk const& col) {
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
                row_group_page_offsets.emplace_back(row_group_page_offsets.back() +
                                                    offset_index.page_locations.size());

                // For all pages in this row group, Get row counts and offsets.
                std::for_each(
                  thrust::counting_iterator<size_t>(0),
                  thrust::counting_iterator(row_group_num_pages),
                  [&](auto const page_idx) {
                    int64_t const first_row_idx =
                      offset_index.page_locations[page_idx].first_row_index;
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
        // Return the computed vectors
        return std::tuple{row_group_page_offsets, page_row_counts, page_row_offsets};
      }();

      CUDF_EXPECTS(page_row_offsets.back() == total_rows, "Mismatch in total rows");
      CUDF_EXPECTS(row_group_page_offsets.back() > 0,
                   "No pages with PageIndex found for the column");

      // Create host columns with page-level min, max values
      auto const total_pages = row_group_page_offsets.back();
      host_column<T> min(total_pages, stream);
      host_column<T> max(total_pages, stream);

      auto page_offset_idx = 0;

      // For each source
      std::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator(row_group_indices.size()),
        [&, row_group_page_offsets = row_group_page_offsets](auto src_idx) {
          // For all row groups in this source
          auto const& rg_indices = row_group_indices[src_idx];
          std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto rg_idx) {
            auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
            auto col =
              std::find_if(row_group.columns.begin(),
                           row_group.columns.end(),
                           [schema_idx](cudf::io::parquet::detail::ColumnChunk const& col) {
                             return col.schema_idx == schema_idx;
                           });

            // No need to check column_index and offset_index again
            if (col != std::end(row_group.columns)) {
              auto const& colchunk               = *col;
              auto const& column_index           = colchunk.column_index.value();
              auto const& offset_index           = colchunk.offset_index.value();
              auto const num_pages_in_colchunk   = column_index.min_values.size();
              auto const page_offset_in_colchunk = row_group_page_offsets[page_offset_idx++];

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
      auto const row_indices =
        [&, page_row_counts = page_row_counts, page_row_offsets = page_row_offsets]() {
          // Move page-level row counts and offsets to device
          auto row_counts  = cudf::detail::make_device_uvector_async(page_row_counts, stream, mr);
          auto row_offsets = cudf::detail::make_device_uvector_async(page_row_offsets, stream, mr);

          // Generate row index mapping
          auto row_indices =
            cudf::detail::make_zeroed_device_uvector_async<cudf::size_type>(total_rows, stream, mr);
          thrust::scatter_if(rmm::exec_policy_nosync(stream),
                             thrust::counting_iterator<size_type>(0),
                             thrust::counting_iterator<size_type>(row_counts.size()),
                             row_offsets.begin(),
                             row_counts.begin(),
                             row_indices.begin());

          // Fill gaps with previous values
          thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                                 row_indices.begin(),
                                 row_indices.end(),
                                 row_indices.begin(),
                                 thrust::maximum<cudf::size_type>());
          return row_indices;
        }();

      if constexpr (not std::is_same_v<T, string_view>) {
        // Lambda function to gather values based on computed indices
        auto const gather_data_and_nullmask = [&](mutable_column_view column,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr) {
          // Buffer for output data
          auto output_data = rmm::device_buffer(cudf::size_of(dtype) * total_rows, stream, mr);

          // Buffer for output bitmask. Set all bits valid
          auto output_bitmask =
            cudf::create_null_mask(total_rows, mask_state::ALL_VALID, stream, mr);

          // For each row index, copy over the min/max page stat value and the bitmask bit from the
          // corresponding page. index is invalid
          thrust::for_each(rmm::exec_policy_nosync(stream),
                           thrust::counting_iterator(0),
                           thrust::counting_iterator(total_rows),
                           [row_indices    = row_indices.begin(),
                            input_data     = column.template begin<T>(),
                            output_data    = reinterpret_cast<T*>(output_data.data()),
                            input_bitmask  = column.null_mask(),
                            output_bitmask = reinterpret_cast<bitmask_type*>(
                              output_bitmask.data())] __device__(auto const row_index) {
                             auto const page_index = row_indices[row_index];
                             // Write output data
                             output_data[row_index] = input_data[page_index];
                             // Write output bitmask
                             if (not bit_is_set(input_bitmask, page_index)) {
                               clear_bit_unsafe(output_bitmask, row_index);
                             }
                           });

          return std::pair{std::move(output_data), std::move(output_bitmask)};
        };

        // Move host columns to device
        auto mincol = min.to_device(dtype, stream, mr);
        auto maxcol = max.to_device(dtype, stream, mr);

        // Convert page-level min and max columns to row-level min and max columns by gathering
        // values based on page-level row offsets
        auto [min_data, min_bitmask] = gather_data_and_nullmask(mincol->mutable_view(), stream, mr);
        auto [max_data, max_bitmask] = gather_data_and_nullmask(maxcol->mutable_view(), stream, mr);

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
      } else {
        auto const gather_strings_children_and_nullmask = [&](host_column<T> const& host_col,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr) {
          // Construct device vectors containing page-level string children and sizes.
          [[maybe_unused]] auto [page_str_chars, page_str_sizes, page_str_offsets] = [&]() {
            auto const& host_strings    = host_col.val;
            auto const total_char_count = std::accumulate(
              host_strings.begin(), host_strings.end(), 0, [](auto sum, auto const& str) {
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

          // Buffer for page-level strings bitmask (input)
          auto input_bitmask = rmm::device_buffer{host_col.null_mask.data(),
                                                  cudf::bitmask_allocation_size_bytes(total_pages),
                                                  stream,
                                                  mr};
          // Buffer for row-level strings bitmask (output). Initialize to all bits set.
          auto output_bitmask =
            cudf::create_null_mask(total_rows, mask_state::ALL_VALID, stream, mr);

          // Gather string sizes and bitmask from page to row level
          thrust::for_each(rmm::exec_policy_nosync(stream),
                           thrust::counting_iterator(0),
                           thrust::counting_iterator(total_rows),
                           [row_indices      = row_indices.begin(),
                            input_str_sizes  = page_str_sizes.begin(),
                            output_str_sizes = row_str_sizes.begin(),
                            input_bitmask  = reinterpret_cast<bitmask_type*>(input_bitmask.data()),
                            output_bitmask = reinterpret_cast<bitmask_type*>(
                              output_bitmask.data())] __device__(auto const row_index) {
                             auto const page_index       = row_indices[row_index];
                             output_str_sizes[row_index] = input_str_sizes[page_index];
                             if (not bit_is_set(input_bitmask, page_index)) {
                               clear_bit_unsafe(output_bitmask, row_index);
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

          // Iterator for input (page-level) strings
          auto src_iter = thrust::make_transform_iterator(
            thrust::make_counting_iterator<std::size_t>(0),
            cuda::proclaim_return_type<char*>(
              [chars       = page_str_chars.begin(),
               offsets     = page_str_offsets.begin(),
               row_indices = row_indices.begin()] __device__(std::size_t index) {
                auto const page_index = row_indices[index];
                return chars + offsets[page_index];
              }));

          // Iterator for output (row-level) strings
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

          // Return row-level (output) strings children and the bitmask
          return std::tuple{
            std::move(row_str_chars), std::move(row_str_offsets), std::move(output_bitmask)};
        };

        auto [min_data, min_offsets, min_bitmask] =
          gather_strings_children_and_nullmask(min, stream, mr);
        auto [max_data, max_offsets, max_bitmask] =
          gather_strings_children_and_nullmask(min, stream, mr);

        // Count nulls in min and max columns
        auto const min_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(min_bitmask.data()), 0, total_rows, stream);
        auto const max_nulls = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(max_bitmask.data()), 0, total_rows, stream);

        // Return min and max device strings columns
        return {
          cudf::make_strings_column(
            total_rows,
            std::make_unique<column>(std::move(min_offsets), rmm::device_buffer{0, stream, mr}, 0),
            std::move(min_data),
            min_nulls,
            std::move(min_bitmask)),
          cudf::make_strings_column(
            total_rows,
            std::make_unique<column>(std::move(max_offsets), rmm::device_buffer{0, stream, mr}, 0),
            std::move(max_data),
            max_nulls,
            std::move(max_bitmask))};
      }
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> aggregate_reader_metadata::filter_data_pages_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<data_type const> output_dtypes,
  cudf::host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
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
            auto col =
              std::find_if(row_group.columns.begin(),
                           row_group.columns.end(),
                           [schema_idx](cudf::io::parquet::detail::ColumnChunk const& col) {
                             return col.schema_idx == schema_idx;
                           });
            return col != per_file_metadata[src_index].row_groups[rg_index].columns.end() and
                   col->offset_index.has_value() and col->column_index.has_value();
          });
        });
    });

  // Return if page index is not present
  if (not has_page_index) { return cudf::make_empty_column(cudf::type_id::BOOL8); }

  // Total number of rows
  auto const total_rows = std::accumulate(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(row_group_indices.size()),
    size_t{0},
    [&](auto sum, auto const src_index) {
      auto const& rg_indices = row_group_indices[src_index];
      return std::accumulate(
        rg_indices.begin(), rg_indices.end(), sum, [&](auto sum, auto const rg_index) {
          CUDF_EXPECTS(sum + per_file_metadata[src_index].row_groups[rg_index].num_rows <=
                         std::numeric_limits<size_type>::max(),
                       "Total rows exceed the maximum value");
          return sum + per_file_metadata[src_index].row_groups[rg_index].num_rows;
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
  cudf::io::parquet::detail::stats_expression_converter const stats_expr{
    filter.value().get(), static_cast<size_type>(output_dtypes.size())};

  // Filter the input table using AST expression and return the (BOOL8) predicate column.
  return cudf::detail::compute_column(stats_table, stats_expr.get_stats_expr().get(), stream, mr);
}

std::vector<std::vector<cudf::io::text::byte_range_info>>
aggregate_reader_metadata::get_filter_columns_data_pages(
  cudf::column_view input_rows,
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  rmm::cuda_stream_view stream) const
{
  // Return if no input row groups
  if (row_group_indices.empty()) { return {}; }

  CUDF_EXPECTS(input_rows.type().id() == cudf::type_id::BOOL8,
               "Input row bitmask should be of type BOOL8");

  auto const total_rows  = input_rows.size();
  auto const num_columns = output_dtypes.size();

  // Compute byte offsets and row offsets for all pages across all filter columns.
  std::vector<std::vector<cudf::io::text::byte_range_info>> all_page_byte_ranges(num_columns);
  std::vector<std::vector<size_type>> page_row_offsets(num_columns);
  std::for_each(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(num_columns),
    [&](auto col_idx) {
      auto const schema_idx = output_column_schemas[col_idx];
      size_type rows_so_far = 0;
      // For all source files
      std::for_each(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator(row_group_indices.size()),
        [&](auto const src_index) {
          // Get all row group indices in the data source
          auto const& rg_indices = row_group_indices[src_index];
          // For all row groups in the source file
          std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto const rg_index) {
            auto const& row_group = per_file_metadata[src_index].row_groups[rg_index];
            auto col =
              std::find_if(row_group.columns.begin(),
                           row_group.columns.end(),
                           [schema_idx](cudf::io::parquet::detail::ColumnChunk const& col) {
                             return col.schema_idx == schema_idx;
                           });
            if (col != std::end(row_group.columns) and col->offset_index.has_value()) {
              auto const& offset_index = col->offset_index.value();
              for (size_t page_idx = 0; page_idx < offset_index.page_locations.size(); ++page_idx) {
                page_row_offsets[col_idx].emplace_back(
                  rows_so_far + offset_index.page_locations[page_idx].first_row_index);
                all_page_byte_ranges[col_idx].emplace_back(
                  offset_index.page_locations[page_idx].offset,
                  offset_index.page_locations[page_idx].compressed_page_size);
              }
            }
            rows_so_far += row_group.num_rows;
          });
        });

      // Insert the last offset of the last page
      page_row_offsets[col_idx].emplace_back(total_rows);
    });

  if (input_rows.is_empty()) { return all_page_byte_ranges; }

  CUDF_EXPECTS(page_row_offsets.back().back() == total_rows, "Mismatch in total rows");

  // Get the validity column bitmask
  auto const host_bitmask = [&] {
    auto const num_bitmasks = num_bitmask_words(input_rows.size());
    if (input_rows.nullable()) {
      return cudf::detail::make_host_vector_sync(
        device_span<bitmask_type const>(input_rows.null_mask(), num_bitmasks), stream);
    } else {
      auto bitmask = cudf::detail::make_host_vector<bitmask_type>(num_bitmasks, stream);
      std::fill(bitmask.begin(), bitmask.end(), ~bitmask_type{0});
      return bitmask;
    }
  }();

  auto const total_input_pages =
    std::accumulate(page_row_offsets.cbegin(),
                    page_row_offsets.cend(),
                    size_t{0},
                    [](auto sum, auto const& offsets) { return sum + offsets.size() - 1; });

  // Create a validity iterator
  auto validity_it = cudf::detail::make_counting_transform_iterator(
    0, [bitmask = host_bitmask.data()](auto bit_index) { return bit_is_set(bitmask, bit_index); });
  // Return only filtered row groups based on predicate
  auto const is_row_required = cudf::detail::make_host_vector_sync(
    device_span<uint8_t const>(input_rows.data<uint8_t>(), input_rows.size()), stream);

  // Return if all rows are required, or all rows are nulls.
  if (input_rows.null_count() == input_rows.size() or
      std::all_of(
        is_row_required.cbegin(), is_row_required.cend(), [](auto i) { return bool(i); })) {
    return all_page_byte_ranges;
  }

  // For all columns, look up which pages contain at least one valid row. i.e. !validity_it[row_idx]
  // or is_row_required[row_idx] satisfies, and add its byte range to the output list of byte ranges
  // for the column.
  auto data_page_bytes = std::vector<std::vector<cudf::io::text::byte_range_info>>();
  data_page_bytes.reserve(num_columns);

  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(num_columns),
                [&](auto col_idx) {
                  // page ranges for the current column
                  auto page_ranges = std::vector<cudf::io::text::byte_range_info>();
                  page_ranges.reserve(page_row_offsets[col_idx].size());
                  // For all rows
                  for (size_type row_idx = 0; row_idx < total_rows; ++row_idx) {
                    // If required row
                    if (not validity_it[row_idx] or is_row_required[row_idx]) {
                      // binary search to find the page index this row_idx belongs to and set the
                      // page index to true page_indices
                      auto const& offsets = page_row_offsets[col_idx];
                      auto const page_itr =
                        std::upper_bound(offsets.cbegin(), offsets.cend(), row_idx);
                      CUDF_EXPECTS(page_itr != offsets.cbegin(), "Invalid page index");
                      auto const page_idx = std::distance(offsets.cbegin(), page_itr) - 1;
                      page_ranges.emplace_back(all_page_byte_ranges[col_idx][page_idx]);
                      // Move row_idx to the last row of the page, so that we don't need to check
                      // the same page again.
                      row_idx = offsets[page_idx + 1] - 1;
                    }
                  }
                  // Move the vector into the global list.
                  data_page_bytes.emplace_back(std::move(page_ranges));
                });

  auto const filtered_num_pages = std::accumulate(
    data_page_bytes.cbegin(),
    data_page_bytes.cend(),
    size_t{0},
    [](auto sum, auto const& data_page_bytes) { return sum + data_page_bytes.size(); });

  CUDF_EXPECTS(filtered_num_pages < total_input_pages,
               "Number of filtered pages must be smaller than total number of input pages");

  std::cout << "Num pages after filter: " << filtered_num_pages << " out of " << total_input_pages
            << std::endl;

  // Return the final lists of data page byte ranges for all columns
  return data_page_bytes;
}

}  // namespace cudf::experimental::io::parquet::detail
