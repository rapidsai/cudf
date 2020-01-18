/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/scatter.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/scatter.cuh>
#include <cudf/strings/string_view.cuh>
#include "thrust/iterator/counting_iterator.h"

#include <thrust/iterator/discard_iterator.h>

namespace cudf {
namespace experimental {
namespace detail {

namespace {

struct dispatch_map_type {
    template <typename map_type, std::enable_if_t<std::is_integral<map_type>::value
     and not std::is_same<map_type, bool8>::value>* = nullptr>
    std::unique_ptr<table> operator()(
      table_view const& source, column_view const& scatter_map,
      table_view const& target, bool check_bounds,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {

    return detail::scatter<map_type>(source,
           scatter_map.begin<map_type>(),
           scatter_map.end<map_type>(),
           target, check_bounds, mr, stream);
  }

  template <typename map_type, std::enable_if_t<not std::is_integral<map_type>::value
      or std::is_same<map_type, bool8>::value>* = nullptr>
  std::unique_ptr<table> operator()(
      table_view const& source, column_view const& scatter_map,
      table_view const& target, bool check_bounds,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    CUDF_FAIL("Scatter map column must be an integral, non-boolean type");
  }

};

template <bool mark_true, typename MapIterator>
__global__ void marking_bitmask_kernel(
    mutable_column_device_view destination,
    MapIterator scatter_map,
    size_type num_scatter_rows)
{
  size_type row = threadIdx.x + blockIdx.x * blockDim.x;

  while (row < num_scatter_rows) {
    size_type const output_row = scatter_map[row];

    if (mark_true){
      destination.set_valid(output_row);
    } else {
      destination.set_null(output_row);
    }

    row += blockDim.x * gridDim.x;
  }
}

template <typename MapIterator>
void scatter_scalar_bitmask(std::vector<std::unique_ptr<scalar>> const& source,
    MapIterator scatter_map, size_type num_scatter_rows,
    std::vector<std::unique_ptr<column>>& target,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream)
{
  constexpr size_type block_size = 256;
  size_type const grid_size = grid_1d(num_scatter_rows, block_size).num_blocks;

  for (size_t i = 0; i < target.size(); ++i) {
    auto const source_is_valid = source[i]->is_valid(stream);
    if (target[i]->nullable() or not source_is_valid) {
      if (not target[i]->nullable()) {
        // Target must have a null mask if the source is not valid
        auto mask = create_null_mask(target[i]->size(), mask_state::ALL_VALID, stream, mr);
        target[i]->set_null_mask(std::move(mask), 0);
      }

      auto target_view = mutable_column_device_view::create(
        target[i]->mutable_view(), stream);

      auto bitmask_kernel = source_is_valid
        ? marking_bitmask_kernel<true, decltype(scatter_map)>
        : marking_bitmask_kernel<false, decltype(scatter_map)>;
      bitmask_kernel<<<grid_size, block_size, 0, stream>>>(
        *target_view, scatter_map, num_scatter_rows);
    }
  }
}

template <typename MapIterator>
struct column_scalar_scatterer {
  template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()(std::unique_ptr<scalar> const& source,
      MapIterator scatter_iter, size_type scatter_rows, column_view const& target,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    auto result = std::make_unique<column>(target, stream, mr);
    auto result_view = result->mutable_view();

    // Use permutation iterator with constant index to dereference scalar data
    auto scalar_impl = static_cast<scalar_type_t<T>*>(source.get());
    auto scalar_iter = thrust::make_permutation_iterator(
      scalar_impl->data(), thrust::make_constant_iterator(0));

    thrust::scatter(rmm::exec_policy(stream)->on(stream), scalar_iter,
      scalar_iter + scatter_rows, scatter_iter,
      result_view.begin<T>());

    return result;
  }

  template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()(std::unique_ptr<scalar> const& source,
      MapIterator scatter_iter, size_type scatter_rows, column_view const& target,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    auto const scalar_impl = static_cast<string_scalar*>(source.get());
    auto const source_view = string_view(scalar_impl->data(), scalar_impl->size());
    auto const begin = thrust::make_constant_iterator(source_view);
    auto const end = begin + scatter_rows;
    return strings::detail::scatter(begin, end, scatter_iter, target, mr, stream);
  }
};

struct scatter_scalar_impl {
  template <typename T, std::enable_if_t<std::is_integral<T>::value
      and not std::is_same<T, bool8>::value>* = nullptr>
  std::unique_ptr<table> operator()(
      std::vector<std::unique_ptr<scalar>> const& source,
      column_view const& indices, table_view const& target, bool check_bounds,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    if (check_bounds) {
      auto const begin = -target.num_rows();
      auto const end = target.num_rows();
      auto bounds = bounds_checker<T>{begin, end};
      CUDF_EXPECTS(indices.size() == thrust::count_if(
        rmm::exec_policy(stream)->on(stream),
        indices.begin<T>(), indices.end<T>(), bounds),
        "Scatter map index out of bounds");
    }

    // Transform negative indices to index + target size
    auto scatter_rows = indices.size();
    auto scatter_iter = thrust::make_transform_iterator(
      indices.begin<T>(), index_converter<T>{target.num_rows()});

    // Second dispatch over data type per column
    auto result = std::vector<std::unique_ptr<column>>(target.num_columns());
    auto scatter_functor = column_scalar_scatterer<decltype(scatter_iter)>{};
    std::transform(source.begin(), source.end(), target.begin(), result.begin(),
      [=](auto const& source_scalar, auto const& target_col) {
        return type_dispatcher(source_scalar->type(), scatter_functor,
          source_scalar, scatter_iter, scatter_rows, target_col, mr, stream);
      });

    scatter_scalar_bitmask(source, scatter_iter, scatter_rows, result, mr, stream);

    return std::make_unique<table>(std::move(result));
  }

  template <typename T, std::enable_if_t<not std::is_integral<T>::value
      or std::is_same<T, bool8>::value>* = nullptr>
  std::unique_ptr<table> operator()(
      std::vector<std::unique_ptr<scalar>> const& source,
      column_view const& indices, table_view const& target, bool check_bounds,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    CUDF_FAIL("Scatter index column must be an integral, non-boolean type");
  }
};

struct scatter_to_tables_impl {
  template <typename T, std::enable_if_t<std::is_integral<T>::value
      and not std::is_same<T, bool8>::value>* = nullptr>
  std::vector<std::unique_ptr<table>> operator()(
      table_view const& input, column_view const& partition_map,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    // Make a mutable copy of the partition map
    auto d_partitions = rmm::device_vector<T>(
      partition_map.begin<T>(), partition_map.end<T>());

    // Initialize gather maps and offsets to sequence
    auto d_gather_maps = rmm::device_vector<size_type>(partition_map.size());
    auto d_offsets = rmm::device_vector<size_type>(partition_map.size());
    thrust::sequence(rmm::exec_policy(stream)->on(stream),
      d_gather_maps.begin(), d_gather_maps.end());
    thrust::sequence(rmm::exec_policy(stream)->on(stream),
      d_offsets.begin(), d_offsets.end());

    // Sort sequence using partition map as key to generate gather maps
    thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
      d_partitions.begin(), d_partitions.end(), d_gather_maps.begin());

    // Reduce unique partitions to extract gather map offsets from sequence
    auto end = thrust::unique_by_key(rmm::exec_policy(stream)->on(stream),
      d_partitions.begin(), d_partitions.end(), d_offsets.begin());

    // Copy partition indices and gather map offsets to host
    auto partitions = thrust::host_vector<T>(d_partitions.begin(), end.first);
    auto offsets = thrust::host_vector<size_type>(d_offsets.begin(), end.second);
    offsets.push_back(partition_map.size());

    CUDF_EXPECTS(partitions.front() >= 0, "Invalid negative partition index");
    auto output = std::vector<std::unique_ptr<table>>(partitions.back() + 1);

    size_t next_partition = 0;
    for (size_t index = 0; index < partitions.size(); ++index) {
      auto const partition = static_cast<size_t>(partitions[index]);

      // Create empty tables for unused partitions
      for (; next_partition < partition; ++next_partition) {
        output[next_partition] = empty_like(input);
      }

      // Gather input rows for the current partition (second dispatch for column types)
      auto const data = d_gather_maps.data().get() + offsets[index];
      auto const size = offsets[index + 1] - offsets[index];
      auto const gather_map = column_view(data_type(INT32), size, data);
      output[partition] = gather(input, gather_map, false, false, false, mr, stream);

      next_partition = partition + 1;
    }

    return output;
  }

  template <typename T, std::enable_if_t<not std::is_integral<T>::value
      or std::is_same<T, bool8>::value>* = nullptr>
  std::vector<std::unique_ptr<table>> operator()(
      table_view const& input, column_view const& partition_map,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    CUDF_FAIL("Partition map column must be an integral, non-boolean type");
  }
};

}  // namespace

std::unique_ptr<table> scatter(
    table_view const& source, column_view const& scatter_map,
    table_view const& target, bool check_bounds,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  CUDF_EXPECTS(source.num_columns() == target.num_columns(),
    "Number of columns in source and target not equal");
  CUDF_EXPECTS(scatter_map.size() <= source.num_rows(),
    "Size of scatter map must be equal to or less than source rows");
  CUDF_EXPECTS(std::equal(source.begin(), source.end(), target.begin(),
    [](auto const& col1, auto const& col2) {
      return col1.type().id() == col2.type().id();
    }), "Column types do not match between source and target");
  CUDF_EXPECTS(scatter_map.has_nulls() == false, "Scatter map contains nulls");

  if (scatter_map.size() == 0) {
    return std::make_unique<table>(target, stream, mr);
  }

  // First dispatch for scatter map index type
  return type_dispatcher(scatter_map.type(), dispatch_map_type{}, source,
    scatter_map, target, check_bounds, mr, stream);
}

std::unique_ptr<table> scatter(
    std::vector<std::unique_ptr<scalar>> const& source, column_view const& indices,
    table_view const& target, bool check_bounds,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  CUDF_EXPECTS(source.size() == static_cast<size_t>(target.num_columns()),
    "Number of columns in source and target not equal");
  CUDF_EXPECTS(std::equal(source.begin(), source.end(), target.begin(),
    [](auto const& scalar, auto const& col) {
      return scalar->type().id() == col.type().id();
    }), "Column types do not match between source and target");
  CUDF_EXPECTS(indices.has_nulls() == false, "indices contains nulls");

  if (indices.size() == 0) {
    return std::make_unique<table>(target, stream, mr);
  }

  // First dispatch for scatter index type
  return type_dispatcher(indices.type(), scatter_scalar_impl{}, source,
    indices, target, check_bounds, mr, stream);
}

std::vector<std::unique_ptr<table>> scatter_to_tables(
    table_view const& input, column_view const& partition_map,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  CUDF_EXPECTS(partition_map.size() <= input.num_rows(), "scatter map larger than input");
  CUDF_EXPECTS(partition_map.has_nulls() == false, "scatter map contains nulls");

  if (partition_map.size() == 0 || input.num_rows() == 0) {
    return std::vector<std::unique_ptr<table>>{};
  }

  // First dispatch for scatter index type
  return type_dispatcher(partition_map.type(), scatter_to_tables_impl{},
    input, partition_map, mr, stream);
}

}  // namespace detail

std::unique_ptr<table> scatter(
    table_view const& source, column_view const& scatter_map,
    table_view const& target, bool check_bounds,
    rmm::mr::device_memory_resource* mr)
{
  return detail::scatter(source, scatter_map, target, check_bounds, mr);
}

std::unique_ptr<table> scatter(
    std::vector<std::unique_ptr<scalar>> const& source, column_view const& indices,
    table_view const& target, bool check_bounds,
    rmm::mr::device_memory_resource* mr)
{
  return detail::scatter(source, indices, target, check_bounds, mr);
}

std::vector<std::unique_ptr<table>> scatter_to_tables(
    table_view const& input, column_view const& partition_map,
    rmm::mr::device_memory_resource* mr)
{
  return detail::scatter_to_tables(input, partition_map, mr);
}

}  // namespace experimental
}  // namespace cudf
