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

#include <cuda/functional>

namespace cudf::io::orc::detail {

namespace {
using cudf::detail::device_2dspan;

#if 0
struct cumulative_row_info {
  size_type row_count;
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
  // to overestimate size somewhat than to underestimate it and potentially generate chunks
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
  size_type const num_rows_per_stripe;
  size_type const num_levels;
  size_type const* const level_num_cols;  // number of columns in each level
  data_type const* const col_types;       // type of each column in each level
  gpu::ColumnDesc const* const chunks;    // data of each column in each level

  __device__ std::size_t get_flattened_index(size_type level, size_type col_idx) const
  {
    return static_cast<std::size_t>(level) * static_cast<std::size_t>(num_levels) +
           static_cast<std::size_t>(col_idx);
  }

  __device__ std::size_t column_size(std::size_t stripe_idx,
                                     size_type level,
                                     size_type col_idx) const
  {
    auto const flattened_idx = get_flattened_index(level, col_idx);
    auto const col_type      = col_types[flattened_idx];
    auto const& chunk        = chunks[flattened_idx];
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
    auto const num_cols = level_num_cols[level];
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

    return {num_rows_per_stripe,
            thrust::reduce(thrust::seq, size_iter, size_iter + num_levels) +
              chunks[stripe_idx].str_bytes};
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

  // Loop over all stripes, compute stripe sizes.
}

}  // namespace cudf::io::orc::detail
