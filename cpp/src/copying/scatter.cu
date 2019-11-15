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
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>

namespace cudf {
namespace experimental {
namespace detail {

namespace {

template <typename T>
std::unique_ptr<column> copy_with_policy(column_view const& original,
    mask_allocation_policy policy, rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  std::unique_ptr<column> copy;
  if (original.nullable() || policy == mask_allocation_policy::RETAIN) {
    // Copy null mask from the original, if it exists
    copy = std::make_unique<column>(original, stream, mr);
  } else {
    // Original doesn't have a null mask, but we may allocate an empty one
    copy = allocate_like(original, original.size(), policy, mr, stream);
    auto copy_view = copy->mutable_view();
    thrust::copy(rmm::exec_policy(stream)->on(stream),
      original.begin<T>(), original.end<T>(), copy_view.begin<T>());
    copy->set_null_count(0);
  }
  return copy;
}

template <typename T>
thrust::device_vector<T> make_gather_map(column_view const& scatter_map,
    size_type rows, cudaStream_t stream)
{
  // Transform negative indices to index + rows
  auto scatter_iter = thrust::make_transform_iterator(
    scatter_map.begin<T>(),
    index_converter<T>{rows});

  static_assert(std::is_signed<T>::value,
    "Need different invalid index if unsigned index types are added");
  auto const invalid_index = static_cast<T>(-1);

  // Convert scatter map to a gather map
  auto gather_map = thrust::device_vector<T>(rows, invalid_index);
  thrust::scatter(rmm::exec_policy(stream)->on(stream),
    thrust::make_counting_iterator<T>(0),
    thrust::make_counting_iterator<T>(scatter_map.size()),
    scatter_iter, gather_map.begin());

  return gather_map;
}

template <bool ignore_out_of_bounds, bool source_nullable, typename MapIterator>
__device__ size_type gather_bitmask_column_inplace(column_device_view& source_col,
    MapIterator gather_map, mutable_column_device_view destination_col)
{
  size_type destination_row_base = blockIdx.x * blockDim.x;
  size_type valid_count_accumulate = 0;

  while (destination_row_base < destination_col.size()) {
    size_type destination_row = destination_row_base + threadIdx.x;

    const bool thread_active = destination_row < destination_col.size();
    size_type source_row = thread_active ? gather_map[destination_row] : 0;

    // Read destination bit if source index is out of bounds
    bool bit_is_valid;
    if (ignore_out_of_bounds && (source_row < 0 || source_row >= source_col.size())) {
      bit_is_valid = thread_active ? destination_col.is_valid(destination_row) : false;
    } else {
      bit_is_valid = source_nullable ? source_col.is_valid_nocheck(source_row) : true;
    }

    // Use ballot to find all valid bits in this warp and create the output bitmask element
    uint32_t const valid_warp = __ballot_sync(0xffffffff, thread_active && bit_is_valid);

    size_type const valid_index = word_index(destination_row);

    // Only one thread writes output
    if (0 == threadIdx.x % warp_size) {
      destination_col.set_mask_word(valid_index, valid_warp);
    }
    valid_count_accumulate += single_lane_block_popc_reduce(valid_warp);
    destination_row_base += blockDim.x * gridDim.x;
  }

  return valid_count_accumulate;
}

template <bool ignore_out_of_bounds, typename MapIterator>
__global__ void gather_bitmask_inplace_kernel(table_device_view source_table,
    MapIterator gather_map, mutable_table_device_view destination_table,
    size_type* valid_counts)
{
  for (size_type i = 0; i < source_table.num_columns(); i++) {
    column_device_view source_col = source_table.column(i);
    mutable_column_device_view destination_col = destination_table.column(i);

    if (destination_col.nullable()) {
      size_type valid_count_accumulate;
      if (source_col.nullable()) {
        valid_count_accumulate = gather_bitmask_column_inplace<ignore_out_of_bounds, true>(
          source_col, gather_map, destination_col);
      } else {
        valid_count_accumulate = gather_bitmask_column_inplace<ignore_out_of_bounds, false>(
          source_col, gather_map, destination_col);
      }
      if (threadIdx.x == 0) {
        atomicAdd(valid_counts + i, valid_count_accumulate);
      }
    }
  }
}

template <typename MapIterator>
void gather_bitmask_inplace(table_view const& source, MapIterator gather_map,
    std::vector<std::unique_ptr<column>>& target, cudaStream_t stream)
{
  auto const device_source = table_device_view::create(source, stream);

  // Make mutable table view from columns
  auto target_views = std::vector<mutable_column_view>(target.size());
  std::transform(target.begin(), target.end(), target_views.begin(),
    [](auto const& col) { return static_cast<mutable_column_view>(*col); });
  auto target_table = mutable_table_view(target_views);
  auto device_target = mutable_table_device_view::create(target_table, stream);

  // Compute block size
  int grid_size, block_size;
  auto bitmask_kernel = gather_bitmask_inplace_kernel<true, decltype(gather_map)>;
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, bitmask_kernel));

  auto valid_counts = thrust::device_vector<size_type>(target.size());
  bitmask_kernel<<<grid_size, block_size, 0, stream>>>(*device_source,
    gather_map, *device_target, valid_counts.data().get());

  // TODO for_each with a zip iterator?
  auto valid_counts_host = thrust::host_vector<size_type>(valid_counts);
  for (size_t i = 0; i < target.size(); ++i) {
    auto const& target_col = target[i];
    if (target_col->nullable()) {
      target_col->set_null_count(target_col->size() - valid_counts_host[i]);
    }
  }
}

template <typename MapIterator>
__global__ void scatter_bitmask_kernel(column_device_view source_col,
    MapIterator scatter_map, size_type scatter_map_size,
    mutable_column_device_view target_col)
{
  size_type row_number = threadIdx.x + blockIdx.x * blockDim.x;

  while (row_number < scatter_map_size) {
    auto const output_row = scatter_map[row_number];
    if (source_col.is_valid(row_number)) {
      target_col.set_valid(output_row);
    } else {
      target_col.set_null(output_row);
    }

    row_number += blockDim.x * gridDim.x;
  }
}

template <typename MapIterator>
void scatter_bitmask(table_view const& source, MapIterator scatter_map,
    size_type scatter_map_size, std::vector<std::unique_ptr<column>>& target,
    cudaStream_t stream)
{
  constexpr size_type block_size{256};
  experimental::detail::grid_1d grid(scatter_map_size, block_size);

  for (size_type i = 0; i < source.num_columns(); ++i) {
    if (target[i]->nullable()) {
      auto source_col = column_device_view::create(source.column(i), stream);
      auto target_col = mutable_column_device_view::create(target[i]->mutable_view(), stream);
      scatter_bitmask_kernel<<<grid.num_blocks, block_size, 0, stream>>>(
          *source_col, scatter_map, scatter_map_size, *target_col);

      // Recompute the null count
      target[i]->null_count();
    }
  }
}

template <typename index_type>
struct column_scatterer {
  template <typename T, std::enable_if_t<is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& source,
      column_view const& scatter_map, column_view const& target,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    // Result column must have a null mask if the source does
    mask_allocation_policy nullable = source.nullable() ?
      mask_allocation_policy::ALWAYS : mask_allocation_policy::RETAIN;
    std::unique_ptr<column> result = copy_with_policy<T>(target, nullable, mr, stream);
    auto result_view = result->mutable_view();

    // Transform negative indices to index + target size
    auto scatter_iter = thrust::make_transform_iterator(
      scatter_map.begin<index_type>(),
      index_converter<index_type>{target.size()});

    // NOTE use source.begin + scatter_map.size rather than end in case the
    // scatter map is smaller than the number of source rows
    thrust::scatter(rmm::exec_policy(stream)->on(stream), source.begin<T>(),
      source.begin<T>() + scatter_map.size(), scatter_iter,
      result_view.begin<T>());

    return result;
  }

  template <typename T, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& source,
      column_view const& scatter_map, column_view const& target,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    CUDF_FAIL("Scatter column type must be fixed width");
  }
};

struct scatter_impl {
  template <typename T, std::enable_if_t<std::is_integral<T>::value
     and not std::is_same<T, bool8>::value>* = nullptr>
  std::unique_ptr<table> operator()(
      table_view const& source, column_view const& scatter_map,
      table_view const& target, bool check_bounds,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    if (check_bounds) {
      auto const begin = -target.num_rows();
      auto const end = target.num_rows();
      auto bounds = bounds_checker<T>{begin, end};
      CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(stream)->on(stream),
        scatter_map.begin<T>(), scatter_map.end<T>(), bounds),
        "Scatter map index out of bounds");
    }

    // TODO create separate streams for each col and then sync with master?
    auto result = std::vector<std::unique_ptr<column>>(target.num_columns());
    std::transform(source.begin(), source.end(), target.begin(), result.begin(),
      [&scatter_map, mr, stream](auto const& source_col, auto const& target_col) {
        return type_dispatcher(source_col.type(), column_scatterer<T>{},
          source_col, scatter_map, target_col, mr, stream);
      });

    constexpr bool use_gather_bitmask = false;
    if (use_gather_bitmask) {
      auto gather_map = make_gather_map<T>(scatter_map, target.num_rows(), stream);
      gather_bitmask_inplace(source, gather_map.begin(), result, stream);
    }
    else {
      // Transform negative indices to index + target size
      auto scatter_iter = thrust::make_transform_iterator(
        scatter_map.begin<T>(), index_converter<T>{target.num_rows()});
      scatter_bitmask(source, scatter_iter, scatter_map.size(), result, stream);
    }

    return std::make_unique<table>(std::move(result));
  }

  template <typename T, std::enable_if_t<not std::is_integral<T>::value
      or std::is_same<T, bool8>::value>* = nullptr>
  std::unique_ptr<table> operator()(
      table_view const& source, column_view const& scatter_map,
      table_view const& target, bool check_bounds,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    CUDF_FAIL("Scatter map column must be an integral, non-boolean type");
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
      CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(stream)->on(stream),
        indices.begin<T>(), indices.end<T>(), bounds),
        "Scatter map index out of bounds");
    }

    // TODO
    return std::make_unique<table>(target, stream, mr);
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
  return type_dispatcher(scatter_map.type(), scatter_impl{}, source,
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

}  // namespace experimental
}  // namespace cudf
