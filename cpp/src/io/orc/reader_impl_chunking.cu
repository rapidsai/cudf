/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>

namespace cudf::io::orc::detail {

namespace {
using cudf::detail::device_2dspan;

#if 1
struct cumulative_row_info {
  uint32_t row_count;
  std::size_t size_bytes;
};

/**
 * @brief Functor which computes the total data size for a given type of cudf column.
 *
 * In the case of strings, the return size does not include the chars themselves. That
 * information is tracked separately (see PageInfo::str_bytes).
 * TODO: Compute str_bytes.
 */
struct column_size_fn {
  static constexpr __device__ std::size_t validity_size(std::size_t num_rows, bool nullable)
  {
    return nullable ? (cudf::util::div_rounding_up_safe(num_rows, std::size_t{32}) * 4) : 0;
  }

  template <typename T>
  __device__ std::size_t operator()(std::size_t num_rows, bool nullable) const
  {
    auto const element_size = sizeof(device_storage_type_t<T>);
    return (element_size * num_rows) + validity_size(num_rows, nullable);
  }
};

template <>
__device__ std::size_t column_size_fn::operator()<list_view>(std::size_t num_rows,
                                                             bool nullable) const
{
  // NOTE: Adding the + 1 offset here isn't strictly correct. There will only be 1 extra offset
  // for the entire column, whereas this is adding an extra offset per stripe. So we will get a
  // small over-estimate of the real size of the order:  # of stripes * 4 bytes. It seems better
  // to overestimate size somewhat than to underestimate it and potentially generate lvl_chunks
  // that are too large.
  return sizeof(size_type) * (num_rows + 1) + validity_size(num_rows, nullable);
}

template <>
__device__ std::size_t column_size_fn::operator()<string_view>(std::size_t num_rows,
                                                               bool nullable) const
{
  // Same as lists.
  return this->operator()<list_view>(num_rows, nullable);
}

template <>
__device__ std::size_t column_size_fn::operator()<struct_view>(std::size_t num_rows,
                                                               bool nullable) const
{
  return validity_size(num_rows, nullable);
}

/**
 * @brief Functor which computes the total output cudf data size for all data in this stripe.
 *
 * For the given stripe, this sums across all nesting levels, and across all columns in each level.
 * Note that sizes of the strings in string columns must be precomputed.
 */
struct stripe_size_fn {
  size_type const num_levels;  // total number of nested levels
  //  size_type const num_stripes;              // total number of all stripes
  uint32_t const* num_rows_per_stripe;          // number of rows in each stripe
  size_type const* const lvl_num_cols;          // number of columns in each level
  size_type const* const lvl_offsets;           // prefix sum of number of columns in each level
  data_type const* const* const lvl_col_types;  // type of each column in each stripe in each level
  gpu::ColumnDesc const* const* const
    lvl_chunks;  // data of each column in each stripe in each level

  __device__ std::size_t column_size(std::size_t stripe_idx,
                                     size_type level,
                                     size_type col_idx) const
  {
    auto const num_cols = lvl_num_cols[level];
    auto const col_type = lvl_col_types[level][col_idx];
    auto const& chunk   = lvl_chunks[level][stripe_idx * num_cols + col_idx];
    // TODO: comput str_bytes.
    return chunk.str_bytes +
           cudf::type_dispatcher(
             col_type, column_size_fn{}, chunk.num_rows, chunk.valid_map_base != nullptr);
  }

  __device__ std::size_t level_size(std::size_t stripe_idx, size_type level) const
  {
    auto const num_cols  = lvl_num_cols[level];
    auto const size_iter = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<std::size_t>(
        [this, num_cols, stripe_idx, level] __device__(size_type col_idx) {
          return this->column_size(stripe_idx, level, col_idx);
        }));
    return thrust::reduce(thrust::seq, size_iter, size_iter + num_cols);
  }

  __device__ cumulative_row_info operator()(std::size_t stripe_idx) const
  {
    // Each level has a different number of columns.
    // Thus, we cannot compute column size by one thread per column.
    // TODO: Implement more efficient algorithm to compute column sizes by one thread per column.
    auto const size_iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<std::size_t>([this, stripe_idx] __device__(size_type level) {
        return this->level_size(stripe_idx, level);
      }));

    return {num_rows_per_stripe[stripe_idx],
            thrust::reduce(thrust::seq, size_iter, size_iter + num_levels)};
  }
};

std::vector<row_chunk> find_splits(host_span<cumulative_row_info const> sizes,
                                   size_type num_rows,
                                   size_t chunk_read_limit)
{
  std::vector<row_chunk> splits;

  uint32_t cur_row_count     = 0;
  int64_t cur_pos            = 0;
  size_t cur_cumulative_size = 0;
  auto const start           = thrust::make_transform_iterator(
    sizes.begin(), [&](auto const& size) { return size.size_bytes - cur_cumulative_size; });
  auto const end = start + static_cast<int64_t>(sizes.size());
  while (cur_row_count < static_cast<uint32_t>(num_rows)) {
    int64_t split_pos = thrust::distance(
      start, thrust::lower_bound(thrust::seq, start + cur_pos, end, chunk_read_limit));

    // If we're past the end, or if the returned bucket is bigger than the chunk_read_limit, move
    // back one.
    if (static_cast<size_t>(split_pos) >= sizes.size() ||
        (sizes[split_pos].size_bytes - cur_cumulative_size > chunk_read_limit)) {
      split_pos--;
    }

    // best-try. if we can't find something that'll fit, we have to go bigger. we're doing this in
    // a loop because all of the cumulative sizes for all the pages are sorted into one big list.
    // so if we had two columns, both of which had an entry {1000, 10000}, that entry would be in
    // the list twice. so we have to iterate until we skip past all of them.  The idea is that we
    // either do this, or we have to call unique() on the input first.
    while (split_pos < (static_cast<int64_t>(sizes.size()) - 1) &&
           (split_pos < 0 || sizes[split_pos].row_count == cur_row_count)) {
      split_pos++;
    }

    auto const start_row = cur_row_count;
    cur_row_count        = sizes[split_pos].row_count;
    splits.emplace_back(row_chunk{start_row, static_cast<size_type>(cur_row_count - start_row)});
    cur_pos             = split_pos;
    cur_cumulative_size = sizes[split_pos].size_bytes;
  }

  return splits;
}

void print_cumulative_row_info(host_span<cumulative_row_info const> sizes,
                               std::string const& label,
                               std::optional<std::vector<row_chunk>> splits = std::nullopt)
{
  if (splits.has_value()) {
    printf("------------\nSplits (start_rows, num_rows): \n");
    for (size_t idx = 0; idx < splits->size(); idx++) {
      printf("{%ld, %ld}\n", splits.value()[idx].start_rows, splits.value()[idx].num_rows);
    }
  }

  printf("------------\nCumulative sizes (row_count, size_bytes): %s\n", label.c_str());
  for (size_t idx = 0; idx < sizes.size(); idx++) {
    printf("{%u, %lu}", sizes[idx].row_count, sizes[idx].size_bytes);

    if (splits.has_value()) {
      // if we have a split at this row count and this is the last instance of this row count
      auto const start = thrust::make_transform_iterator(
        splits->begin(), [](auto const& i) { return i.start_rows; });
      auto const end         = start + splits->size();
      auto const split       = std::find(start, end, sizes[idx].row_count);
      auto const split_index = [&]() -> int {
        if (split != end &&
            ((idx == sizes.size() - 1) || (sizes[idx + 1].row_count > sizes[idx].row_count))) {
          return static_cast<int>(std::distance(start, split));
        }
        return idx == 0 ? 0 : -1;
      }();
      if (split_index >= 0) {
        printf(" <-- split {%lu, %lu}",
               splits.value()[split_index].start_rows,
               splits.value()[split_index].num_rows);
      }
    }

    printf("\n");
  }
}

#endif

struct cumulative_row_sum {
  __device__ cumulative_row_info operator()(cumulative_row_info const& a,
                                            cumulative_row_info const& b) const
  {
    return cumulative_row_info{a.row_count + b.row_count, a.size_bytes + b.size_bytes};
  }
};

}  // namespace

void reader::impl::compute_chunk_ranges()
{
  // If there is no limit on the output size, we just read everything.
  if (_chunk_read_info.chunk_size_limit == 0) {
    _chunk_read_info.chunks = {{_file_itm_data.rows_to_skip, _file_itm_data.rows_to_read}};
    return;
  }

  // Compute string sizes for all strings column.
  // TODO

  //  auto const& selected_stripes = _file_itm_data.selected_stripes;
  auto const num_stripes = _file_itm_data.stripe_sizes.size();
  auto const num_levels  = static_cast<size_type>(_selected_columns.num_levels());

  _file_itm_data.lvl_num_cols.host_to_device_async(_stream);
  _file_itm_data.lvl_offsets.host_to_device_async(_stream);
  _file_itm_data.stripe_sizes.host_to_device_async(_stream);

  hostdevice_vector<data_type const*> lvl_col_types(0, num_levels, _stream);
  hostdevice_vector<gpu::ColumnDesc const*> lvl_chunks(0, num_levels, _stream);
  for (size_type level = 0; level < num_levels; ++level) {
    auto& col_types = _file_itm_data.lvl_col_types[level];
    col_types.host_to_device_async(_stream);
    lvl_col_types.push_back(col_types.d_begin());
    lvl_chunks.push_back(_file_itm_data.lvl_chunks[level].base_device_ptr());
  }
  lvl_col_types.host_to_device_async(_stream);
  lvl_chunks.host_to_device_async(_stream);

  hostdevice_vector<cumulative_row_info> stripe_size_bytes(num_stripes, _stream);
  thrust::transform(rmm::exec_policy(_stream),
                    thrust::make_counting_iterator(std::size_t{0}),
                    thrust::make_counting_iterator(num_stripes),
                    stripe_size_bytes.d_begin(),
                    stripe_size_fn{num_levels,
                                   _file_itm_data.stripe_sizes.d_begin(),
                                   _file_itm_data.lvl_num_cols.d_begin(),
                                   _file_itm_data.lvl_offsets.d_begin(),
                                   lvl_col_types.d_begin(),
                                   lvl_chunks.d_begin()});

  thrust::inclusive_scan(rmm::exec_policy(_stream),
                         stripe_size_bytes.d_begin(),
                         stripe_size_bytes.d_end(),
                         stripe_size_bytes.d_begin(),
                         cumulative_row_sum{});

  stripe_size_bytes.device_to_host_sync(_stream);

  _chunk_read_info.chunks =
    find_splits(stripe_size_bytes,
                _file_itm_data.rows_to_read, /*_chunk_read_info.chunk_size_limit*/
                500);

  std::cout << "  total rows: " << _file_itm_data.rows_to_read << std::endl;
  print_cumulative_row_info(stripe_size_bytes, "  ", _chunk_read_info.chunks);
}

}  // namespace cudf::io::orc::detail
