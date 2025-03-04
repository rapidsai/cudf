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
      // #if 0
      //   TODO: For non string columns, we can do this with GPU.
      //   We can for string columns as well but the algorithm needs modification
      if constexpr (not std::is_same_v<T, string_view>) {
        // Compute page offsets
        auto const [page_offsets, page_row_counts, page_row_offsets] = [&]() {
          std::vector<size_type> page_offsets{0};
          std::vector<size_type> page_row_counts;
          std::vector<size_type> page_row_offsets{0};
          std::for_each(
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator(row_group_indices.size()),
            [&](auto src_idx) {
              auto const& rg_indices = row_group_indices[src_idx];
              std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto rg_idx) {
                auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
                auto col =
                  std::find_if(row_group.columns.begin(),
                               row_group.columns.end(),
                               [schema_idx](cudf::io::parquet::detail::ColumnChunk const& col) {
                                 return col.schema_idx == schema_idx;
                               });
                if (col != std::end(row_group.columns) and col->column_index.has_value()) {
                  CUDF_EXPECTS(col->offset_index.has_value(),
                               "Both column and offset index must be present");
                  CUDF_EXPECTS(col->column_index.value().min_values.size() ==
                                 col->column_index.value().max_values.size(),
                               "page min and max values should be of same size");
                  CUDF_EXPECTS(col->column_index.value().min_values.size() ==
                                 col->offset_index.value().page_locations.size(),
                               "mismatch between size of min/max page values and the size of page "
                               "locations");

                  auto const& offset_index = col->offset_index.value();
                  auto const& column_index = col->column_index.value();
                  auto const num_pages     = col->offset_index.value().page_locations.size();

                  page_offsets.emplace_back(page_offsets.back() +
                                            offset_index.page_locations.size());

                  for (size_t page_idx = 0; page_idx < num_pages; ++page_idx) {
                    int64_t const first_row_idx =
                      offset_index.page_locations[page_idx].first_row_index;

                    int64_t const last_row_idx =
                      (page_idx < num_pages - 1)
                        ? offset_index.page_locations[page_idx + 1].first_row_index
                        : row_group.num_rows;

                    page_row_counts.emplace_back(last_row_idx - first_row_idx);
                    page_row_offsets.emplace_back(page_row_offsets.back() + page_row_counts.back());
                  }
                }
              });
            });
          return std::tuple{page_offsets, page_row_counts, page_row_offsets};
        }();

        auto const num_pages = page_offsets.back();
        CUDF_EXPECTS(num_pages > 0, "No pages with PageIndex found for the column");
        CUDF_EXPECTS(page_row_offsets.back() == total_rows, "Mismatch in total rows");

        host_column<T> min(num_pages, stream);
        host_column<T> max(num_pages, stream);

        // For each source
        std::for_each(
          thrust::counting_iterator<size_t>(0),
          thrust::counting_iterator(row_group_indices.size()),
          [&, page_offsets = page_offsets](auto src_idx) {
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
              if (col != std::end(row_group.columns) and col->column_index.has_value()) {
                auto const& colchunk             = *col;
                auto const& column_index         = colchunk.column_index.value();
                auto const& offset_index         = colchunk.offset_index.value();
                auto const num_pages_in_colchunk = column_index.min_values.size();
                auto const page_offset_in_colchunk =
                  page_offsets[src_idx * row_group_indices.size() + rg_idx];

                // For all pages in this chunk
                std::for_each(
                  thrust::counting_iterator<size_t>(0),
                  thrust::counting_iterator(num_pages_in_colchunk),
                  [&](auto page_idx) {
                    // To support deprecated min, max fields.
                    auto const& min_value = column_index.min_values[page_idx];
                    auto const& max_value = column_index.min_values[page_idx];
                    // translate binary data to Type then to <T>
                    min.set_index(
                      page_offset_in_colchunk + page_idx, min_value, colchunk.meta_data.type);
                    max.set_index(
                      page_offset_in_colchunk + page_idx, max_value, colchunk.meta_data.type);
                  });
              }
            });
          });

        auto mincol = min.to_device(dtype, stream, mr);
        auto maxcol = max.to_device(dtype, stream, mr);

        // auto null_mask = std::move(contents.null_mask);
        auto counts  = cudf::detail::make_device_uvector_async(page_row_counts, stream, mr);
        auto offsets = cudf::detail::make_device_uvector_async(page_row_offsets, stream, mr);

        // Generate index mapping
        auto indices =
          cudf::detail::make_zeroed_device_uvector_async<cudf::size_type>(num_pages, stream, mr);
        thrust::scatter_if(rmm::exec_policy_nosync(stream),
                           thrust::counting_iterator<size_type>(0),
                           thrust::counting_iterator<size_type>(counts.size()),
                           offsets.begin(),
                           counts.begin(),
                           offsets.begin());

        // Fill gaps with previous values
        thrust::inclusive_scan(rmm::exec_policy_nosync(stream),
                               indices.begin(),
                               indices.end(),
                               indices.begin(),
                               thrust::maximum<cudf::size_type>());

        // Gather values based on computed indices
        // Column contents
        auto const gather_data_and_nullmask = [&](mutable_column_view column,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr) {
          auto output_data = rmm::device_buffer(cudf::size_of(dtype) * total_rows, stream, mr);
          auto data_ptr    = reinterpret_cast<T*>(output_data.data());
          thrust::gather(rmm::exec_policy_nosync(stream),
                         indices.begin(),
                         indices.end(),
                         column.template begin<T>(),
                         data_ptr);

          // Gather values based on computed indices
          auto bitmask        = column.null_mask();
          auto output_bitmask = rmm::device_buffer(sizeof(bitmask_type) * total_rows, stream, mr);
          auto bitmask_ptr    = reinterpret_cast<bitmask_type*>(output_data.data());
          thrust::gather(
            rmm::exec_policy_nosync(stream), indices.begin(), indices.end(), bitmask, bitmask_ptr);

          return std::pair{std::move(output_data), std::move(output_bitmask)};
        };

        auto [min_data, min_bitmask] = gather_data_and_nullmask(mincol->mutable_view(), stream, mr);
        auto min_nulls               = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(min_bitmask.data()), 0, num_pages, stream);

        auto [max_data, max_bitmask] = gather_data_and_nullmask(maxcol->mutable_view(), stream, mr);
        auto max_nulls               = cudf::detail::null_count(
          reinterpret_cast<bitmask_type*>(max_bitmask.data()), 0, num_pages, stream);

        return {std::make_unique<column>(
                  dtype, total_rows, std::move(min_data), std::move(min_bitmask), min_nulls),
                std::make_unique<column>(
                  dtype, total_rows, std::move(max_data), std::move(max_bitmask), max_nulls)};
      } else
      // #endif
      {
        // Idea: Do we even need to do all this for `cudf::string_view`? No because stats are
        // useless for strings and we end up converting everything to a `cudf::string_view{nullptr,
        // 0}` anyway. So, let's just do that for all rows at once and skip the rest of the code
        // here.
        host_column<T> min(total_rows, stream);
        host_column<T> max(total_rows, stream);
        size_type stats_idx = 0;

        for (size_t src_idx = 0; src_idx < row_group_indices.size(); ++src_idx) {
          for (auto const rg_idx : row_group_indices[src_idx]) {
            auto const& row_group = per_file_metadata[src_idx].row_groups[rg_idx];
            auto col =
              std::find_if(row_group.columns.begin(),
                           row_group.columns.end(),
                           [schema_idx](cudf::io::parquet::detail::ColumnChunk const& col) {
                             return col.schema_idx == schema_idx;
                           });
            if (col != std::end(row_group.columns) and col->column_index.has_value()) {
              auto const& colchunk     = *col;
              auto const& column_index = colchunk.column_index.value();
              auto const& offset_index = colchunk.offset_index.value();

              CUDF_EXPECTS(column_index.min_values.size() == column_index.max_values.size(),
                           "page min and max values should be of same size");
              CUDF_EXPECTS(
                column_index.min_values.size() == offset_index.page_locations.size(),
                "mismatch between size of min/max page values and the size of page locations");
              for (size_t page_idx = 0; page_idx < column_index.min_values.size(); ++page_idx) {
                // To support deprecated min, max fields.
                auto const& min_value       = column_index.min_values[page_idx];
                auto const& max_value       = column_index.min_values[page_idx];
                int64_t const first_row_idx = offset_index.page_locations[page_idx].first_row_index;
                int64_t const last_row_idx =
                  (page_idx < column_index.min_values.size() - 1)
                    ? offset_index.page_locations[page_idx + 1].first_row_index
                    : row_group.num_rows;
                for (int64_t i = first_row_idx; i < last_row_idx; ++i) {
                  // translate binary data to Type then to <T>
                  min.set_index(stats_idx + i, min_value, colchunk.meta_data.type);
                  max.set_index(stats_idx + i, max_value, colchunk.meta_data.type);
                }
              }
            } else {
              for (int64_t i = 0; i < row_group.num_rows; ++i) {
                // Marking it null, if column present in row group
                min.set_index(stats_idx + i, std::nullopt, {});
                max.set_index(stats_idx + i, std::nullopt, {});
              }
            }
            stats_idx += row_group.num_rows;
          }
        }
        return {min.to_device(dtype, stream, mr), max.to_device(dtype, stream, mr)};
      }
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> aggregate_reader_metadata::filter_data_pages_with_stats(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  host_span<data_type const> output_dtypes,
  host_span<int const> output_column_schemas,
  std::optional<std::reference_wrapper<ast::expression const>> filter,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  // Converts Column chunk statistics to a table
  // where min(col[i]) = columns[i*2], max(col[i])=columns[i*2+1]
  // For each column, it contains total number of rows from all row groups.
  size_t const total_rows = std::accumulate(
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator(row_group_indices.size()),
    size_t{0},
    [&](auto sum, auto const src_index) {
      auto const& rg_indices = row_group_indices[src_index];
      return std::accumulate(
        rg_indices.begin(), rg_indices.end(), sum, [&](auto sum, auto const& rg_index) {
          CUDF_EXPECTS(sum + per_file_metadata[src_index].row_groups[rg_index].num_rows <=
                         std::numeric_limits<size_type>::max(),
                       "Total rows exceed the maximum value");
          return sum + per_file_metadata[src_index].row_groups[rg_index].num_rows;
        });
    });

  std::vector<std::unique_ptr<column>> columns;
  page_stats_caster const stats_col{
    static_cast<size_type>(total_rows), per_file_metadata, row_group_indices};
  for (size_t col_idx = 0; col_idx < output_dtypes.size(); col_idx++) {
    auto const schema_idx = output_column_schemas[col_idx];
    auto const& dtype     = output_dtypes[col_idx];
    // Only comparable types except fixed point are supported.
    if (cudf::is_compound(dtype) && dtype.id() != cudf::type_id::STRING) {
      // placeholder only for unsupported types.
      columns.push_back(cudf::make_numeric_column(
        data_type{cudf::type_id::BOOL8}, total_rows, rmm::device_buffer{}, 0, stream, mr));
      columns.push_back(cudf::make_numeric_column(
        data_type{cudf::type_id::BOOL8}, total_rows, rmm::device_buffer{}, 0, stream, mr));
      continue;
    }
    auto [min_col, max_col] =
      cudf::type_dispatcher<dispatch_storage_type>(dtype, stats_col, schema_idx, dtype, stream, mr);
    columns.push_back(std::move(min_col));
    columns.push_back(std::move(max_col));
  }
  auto stats_table = cudf::table(std::move(columns));

  // Converts AST to StatsAST with reference to min, max columns in above `stats_table`.
  cudf::io::parquet::detail::stats_expression_converter const stats_expr{
    filter.value().get(), static_cast<size_type>(output_dtypes.size())};

  // Filter the input table using AST expression
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
  if (row_group_indices.empty()) { return {}; }

  CUDF_EXPECTS(input_rows.type().id() == cudf::type_id::BOOL8,
               "Input row bitmask should be of type BOOL8");

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

  auto validity_it = cudf::detail::make_counting_transform_iterator(
    0, [bitmask = host_bitmask.data()](auto bit_index) { return bit_is_set(bitmask, bit_index); });
  // Return only filtered row groups based on predicate
  auto const is_row_required = cudf::detail::make_host_vector_sync(
    device_span<uint8_t const>(input_rows.data<uint8_t>(), input_rows.size()), stream);

  // Return if all are required, or all are nulls.
  // FIXME: Revisit this.
  // if (input_rows.null_count() == input_rows.size() or
  //    std::all_of(
  //      is_row_required.cbegin(), is_row_required.cend(), [](auto i) { return bool(i); })) {
  //  return {};
  //}

  std::vector<std::vector<size_type>> page_row_offsets(output_dtypes.size());
  std::vector<std::vector<cudf::io::text::byte_range_info>> page_byte_ranges(output_dtypes.size());
  size_type curr_rows = 0;

  // For all columns in the output_dtypes
  for (size_t col_idx = 0; col_idx < output_dtypes.size(); col_idx++) {
    auto const schema_idx = output_column_schemas[col_idx];
    curr_rows             = 0;
    size_t num_pages      = 0;
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
          auto col              = std::find_if(row_group.columns.begin(),
                                  row_group.columns.end(),
                                  [schema_idx](cudf::io::parquet::detail::ColumnChunk const& col) {
                                    return col.schema_idx == schema_idx;
                                  });
          if (col != std::end(row_group.columns) and col->column_index.has_value()) {
            auto const& colchunk     = *col;
            auto const& column_index = colchunk.column_index.value();
            auto const& offset_index = colchunk.offset_index.value();
            for (size_t page_idx = 0; page_idx < offset_index.page_locations.size(); ++page_idx) {
              page_row_offsets[col_idx].push_back(
                curr_rows + offset_index.page_locations[page_idx].first_row_index);
              page_byte_ranges[col_idx].push_back(cudf::io::text::byte_range_info{
                offset_index.page_locations[page_idx].offset,
                offset_index.page_locations[page_idx].compressed_page_size});
            }
          }
          curr_rows += row_group.num_rows;
          num_pages += col->offset_index.value().page_locations.size();
        });
      });

    // Insert the last offset of the last page
    page_row_offsets[col_idx].push_back(curr_rows);
  }

  auto const total_rows = curr_rows;
  auto data_page_bytes  = std::vector<std::vector<cudf::io::text::byte_range_info>>();
  data_page_bytes.reserve(output_dtypes.size());

  // For all rows in total rows, if (not validity_it[row_idx] or is_row_required[row_idx]), then
  // look which page it belongs to and get its byte range.
  // For all columns
  for (size_t col_idx = 0; col_idx < output_dtypes.size(); col_idx++) {
    // page ranges for the current column
    auto page_ranges = std::vector<cudf::io::text::byte_range_info>();
    page_ranges.reserve(page_row_offsets[col_idx].size());
    // For all rows
    for (size_type row_idx = 0; row_idx < total_rows; ++row_idx) {
      // If required row
      if (not validity_it[row_idx] or is_row_required[row_idx]) {
        // binary search to find the page index this row_idx belongs to and set the page index to
        // true page_indices
        auto const& offsets = page_row_offsets[col_idx];
        auto const page_itr = std::upper_bound(offsets.cbegin(), offsets.cend(), row_idx);
        CUDF_EXPECTS(page_itr != offsets.cbegin(), "Invalid page index");
        auto const page_idx = std::distance(offsets.cbegin(), page_itr) - 1;
        page_ranges.push_back(page_byte_ranges[col_idx][page_idx]);
        // Move row_idx to the last row of the page, so that we don't need to check the same page
        // again.
        row_idx = offsets[page_idx + 1] - 1;
      }
    }
    // Move the vector into the global list.
    data_page_bytes.push_back(std::move(page_ranges));
  }

  return data_page_bytes;
}

}  // namespace cudf::experimental::io::parquet::detail
