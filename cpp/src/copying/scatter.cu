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
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>

namespace cudf {
namespace experimental {

namespace {

template <typename T>
__global__
void invert_map(mutable_column_device_view gather_map, column_device_view scatter_map)
{
  auto source = threadIdx.x + blockIdx.x * blockDim.x;
  if (source < scatter_map.size()) {
    T dest = scatter_map.element<T>(source);
    if (dest < 0) {
      dest += gather_map.size();
    }
    if (dest >= 0 && dest < gather_map.size()) {
      gather_map.element<T>(dest) = static_cast<T>(source);
    }
  }
}

struct scatter_impl {
  template <typename T, std::enable_if_t<std::is_integral<T>::value
     and not std::is_same<T, bool8>::value>* = nullptr>
  std::unique_ptr<table> operator()(
      table_view const& source, column_view const& scatter_map,
      table_view const& target, bool check_bounds,
      rmm::mr::device_memory_resource* mr, cudaStream_t stream)
  {
    // Negative values are used to indicate unmodified rows in the gather map,
    // so assert against unsigned integer types added in the future
    static_assert(std::is_signed<T>::value,
      "Need special case to handle unsigned index types");

    if (check_bounds) {
      auto const begin = -target.num_rows();
      auto const end = target.num_rows();
      auto bounds = detail::bounds_checker<T>{begin, end};
      CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(stream)->on(stream),
        scatter_map.begin<T>(), scatter_map.end<T>(), bounds),
        "Scatter map index out of bounds");
    }

    // Allocate gather map initialized with negative indices
    auto const default_value = static_cast<T>(-1);
    auto gather_map = detail::allocate_like(scatter_map, target.num_rows(),
      mask_allocation_policy::NEVER, mr, stream);
    auto gather_map_view = gather_map->mutable_view();
    thrust::fill(rmm::exec_policy(stream)->on(stream),
      gather_map_view.begin<T>(), gather_map_view.end<T>(), default_value);

    // TODO replace invert_map with thrust::scatter?
    // Invert the scatter map into the gather map
    auto grid = detail::grid_1d(source.num_rows(), 256);
    auto gather_map_device = mutable_column_device_view::create(gather_map_view, stream);
    auto scatter_map_device = column_device_view::create(scatter_map, stream);
    invert_map<T><<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(
      *gather_map_device, *scatter_map_device);

    // TODO
    return std::make_unique<table>(target, stream, mr);
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
    // Negative values are used to indicate unmodified rows in the gather map,
    // so assert against unsigned integer types added in the future
    static_assert(std::is_signed<T>::value,
      "Need special case to handle unsigned index types");

    if (check_bounds) {
      auto const begin = -target.num_rows();
      auto const end = target.num_rows();
      auto bounds = detail::bounds_checker<T>{begin, end};
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

namespace detail {

std::unique_ptr<table> scatter(
    table_view const& source, column_view const& scatter_map,
    table_view const& target, bool check_bounds,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
  CUDF_EXPECTS(source.num_columns() == target.num_columns(),
    "Number of columns in source and target not equal");
  CUDF_EXPECTS(std::equal(source.begin(), source.end(), target.begin(),
    [](auto const& col1, auto const& col2) {
      return col1.type().id() == col2.type().id();
    }), "Column types do not match between source and target");
  CUDF_EXPECTS(scatter_map.has_nulls() == false, "scatter_map contains nulls");

  // TODO need to assert that scatter_map.size() == source.num_rows()?

  if (scatter_map.size() == 0) {
    return std::make_unique<table>(target, stream, mr);
  }

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
