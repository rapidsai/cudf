/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/scatter.cuh>
#include <cudf/strings/string_view.cuh>
#include <memory>

namespace cudf {
namespace experimental {
namespace detail {

namespace {
template <typename T, typename MapIterator>
rmm::device_vector<T> make_gather_map(MapIterator scatter_map_begin,
    MapIterator scatter_map_end, size_type gather_rows,
    cudaStream_t stream)
{
  static_assert(std::is_signed<T>::value,
    "Need different invalid index if unsigned index types are added");
  auto const invalid_index = static_cast<T>(-1);

  // Convert scatter map to a gather map
  auto gather_map = rmm::device_vector<T>(gather_rows, invalid_index);
  thrust::scatter(rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<T>(0),
    thrust::make_counting_iterator<T>(std::distance(scatter_map_begin, scatter_map_end)),
    scatter_map_begin,
    gather_map.begin());

  return gather_map;
}

template <typename MapIterator>
void gather_bitmask(table_view const& source, MapIterator gather_map,
    std::vector<std::unique_ptr<column>>& target,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  // Create null mask if source is nullable but target is not
  for (size_t i = 0; i < target.size(); ++i) {
    if (source.column(i).nullable() and not target[i]->nullable()) {
      auto mask = create_null_mask(target.size(), mask_state::ALL_VALID, stream, mr);
      target[i]->set_null_mask(std::move(mask), 0);
    }
  }

  auto const device_source = table_device_view::create(source, stream);

  // Make device array of target bitmask pointers
  thrust::host_vector<bitmask_type*> target_masks(target.size());
  std::transform(target.begin(), target.end(), target_masks.begin(),
    [](auto const& col) { return col->mutable_view().null_mask(); });
  rmm::device_vector<bitmask_type*> d_target_masks(target_masks);
  auto target_rows = target.front()->size();

  // Compute block size
  auto bitmask_kernel = gather_bitmask_kernel<true, decltype(gather_map)>;
  constexpr size_type block_size = 256;
  size_type const grid_size = grid_1d(target_rows, block_size).num_blocks;

  auto d_valid_counts = rmm::device_vector<size_type>(target.size());
  bitmask_kernel<<<grid_size, block_size, 0, stream>>>(*device_source,
    gather_map, d_target_masks.data().get(), target_rows, d_valid_counts.data().get());

  // Copy the valid counts into each column
  auto const valid_counts = thrust::host_vector<size_type>(d_valid_counts);
  size_t index = 0;
  for (auto& target_col : target) {
    if (target_col->nullable()) {
      auto const null_count = target_rows - valid_counts[index++];
      target_col->set_null_count(null_count);
    }
  }
}

template <typename MapIterator>
struct column_scatterer {
  template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& source,
      MapIterator scatter_map_begin, MapIterator scatter_map_end, column_view const& target,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    auto result = std::make_unique<column>(target, stream, mr);
    auto result_view = result->mutable_view();

    // NOTE use source.begin + scatter rows rather than source.end in case the
    // scatter map is smaller than the number of source rows
    thrust::scatter(rmm::exec_policy(stream)->on(stream), source.begin<T>(),
      source.begin<T>() + std::distance(scatter_map_begin, scatter_map_end), 
      scatter_map_begin,
      result_view.begin<T>());

    return result;
  }

  template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& source,
      MapIterator scatter_map_begin, MapIterator scatter_map_end, column_view const& target,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    using strings::detail::create_string_vector_from_column;
    auto const source_vector = create_string_vector_from_column(source, stream);
    auto const begin = source_vector.begin();
    auto const end = begin + std::distance(scatter_map_begin, scatter_map_end);
    return strings::detail::scatter(begin, end, scatter_map_begin, target, mr, stream);
  }
};

} //namespace

/**
 * @brief Scatters the rows of the source table into a copy of the target table
 * according to a scatter map.
 *
 * Scatters values from the source table into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_begin[i]` of the destination table gets row
 * `i` of the source table. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of columns in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * @throws cudf::logic_error if scatter_map.size() > source.num_rows()
 *
 * @param[in] source The input columns containing values to be scattered into the
 * target columns
 * @param[in] scatter_map_begin Beginning of iterator range of integer indices that map the rows in the
 * source columns to rows in the target columns
 * @param[in] scatter_map_end End of iterator range of integer indices that map the rows in the
 * source columns to rows in the target columns
 * @param[in] target The set of columns into which values from the source_table
 * are to be scattered
 * @param[in] check_bounds Optionally perform bounds checking on the values of
 * `scatter_map` and throw an error if any of its values are out of bounds.
 * @param[in] mr The resource to use for all allocations
 * @param[in] stream The stream to use for CUDA operations
 *
 * @return Result of scattering values from source to target
 **/
template <typename T, typename MapIterator>
std::unique_ptr<table>
scatter(table_view const& source, MapIterator scatter_map_begin,
    MapIterator scatter_map_end, table_view const& target,
    bool check_bounds = false,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0) {

    CUDF_EXPECTS(std::distance(scatter_map_begin, scatter_map_end) <= source.num_rows(),
            "scatter map size should be less than of equal to number of rows in source");

    auto result = std::vector<std::unique_ptr<column>>(target.num_columns());
    auto scatter_functor = column_scatterer<MapIterator>{};
    std::transform(source.begin(), source.end(), target.begin(), result.begin(),
      [=](auto const& source_col, auto const& target_col) {
        return type_dispatcher(source_col.type(), scatter_functor,
          source_col, scatter_map_begin, scatter_map_end, target_col, mr, stream);
      });

    auto gather_map = make_gather_map<T>(scatter_map_begin, scatter_map_end,
      target.num_rows(), stream);
    gather_bitmask(source, gather_map.begin(), result, mr, stream);

    return std::make_unique<table>(std::move(result));
}
} //namespace detail
} //namespace experimental
} //namespace detail
