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
  uint32_t const* num_rows_per_stripe;    // number of rows in each stripe
  size_type const* const lvl_num_cols;    // number of columns in each level
  size_type const* const lvl_offsets;     // prefix sum of number of columns in each level
  data_type const** const lvl_col_types;  // type of each column in each stripe in each level
  gpu::ColumnDesc const* const* const
    lvl_chunks;  // data of each column in each stripe in each level

  __device__ std::size_t column_size(std::size_t stripe_idx,
                                     size_type level,
                                     size_type col_idx) const
  {
    // TODO
    auto const start_col_idx = lvl_offsets[level];
    auto const col_type      = lvl_col_types[start_col_idx + col_idx];
    auto const& chunk        = lvl_chunks[level][todo + col_idx];
    return cudf::type_dispatcher(
      col_type, column_size_fn{}, chunk.num_rows, chunk.valid_map_base != nullptr);
  }

  __device__ std::size_t level_size(std::size_t stripe_idx, size_type level) const
  {
    auto const size_iter = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<std::size_t>(
        [this, stripe_idx, level] __device__(size_type column_idx) {
          return this->column_size(stripe_idx, level, column_idx);
        }));
    auto const num_cols = lvl_num_cols[level];
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
            thrust::reduce(thrust::seq, size_iter, size_iter + num_levels) +
              lvl_chunks[stripe_idx].str_bytes};
  }
};

std::vector<row_range> find_splits(std::vector<cumulative_row_info> const& sizes,
                                   size_t num_rows,
                                   size_t chunk_read_limit)
{
  //
}

#endif

}  // namespace

void reader::impl::compute_chunk_ranges()
{
  // If there is no limit on the output size, we just read everything.
  if (_chunk_read_info.chunk_size_limit == 0) {
    _chunk_read_info.chunk_ranges = {
      row_range{_file_itm_data.rows_to_skip, _file_itm_data.rows_to_read}};
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

  hostdevice_vector<data_type const*> lvl_col_types(num_levels, _stream);
  hostdevice_vector<gpu::ColumnDesc const*> lvl_chunks(num_levels, _stream);
  for (std::size_t level = 0; level < num_levels; ++level) {
    auto& col_types = _file_itm_data.lvl_col_types[level];
    col_types.host_to_device_async(_stream);
    lvl_col_types.push_back(col_types.d_begin());
    lvl_chunks.push_back(_file_itm_data.lvl_chunks[level].base_device_ptr());
  }
  lvl_col_types.host_to_device_async(_stream);
  lvl_chunks.host_to_device_async(_stream);

  rmm::device_uvector<std::size_t> stripe_size_bytes(num_stripes, _stream);
  thrust::transform(rmm::exec_policy(_stream),
                    thrust::make_counting_iterator(std::size_t{0}),
                    thrust::make_counting_iterator(num_stripes),
                    stripe_size_bytes.begin(),
                    stripe_size_fn{num_levels,
                                   _file_itm_data.stripe_sizes.d_begin(),
                                   _file_itm_data.lvl_num_cols.d_begin(),
                                   _file_itm_data.lvl_offsets.d_begin(),
                                   lvl_col_types.d_begin(),
                                   lvl_chunks.d_begin()});
}

}  // namespace cudf::io::orc::detail
