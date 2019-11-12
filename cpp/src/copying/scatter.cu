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
#include <cudf/detail/scatter.hpp>

namespace cudf {
namespace experimental {

namespace {

struct scatter_impl {
  template <typename map_type, std::enable_if_t<std::is_integral<map_type>::value
    and not std::is_same<map_type, cudf::experimental::bool8>::value>* = nullptr>
  std::unique_ptr<table> operator()(
      table_view const& source, column_view const& scatter_map,
      table_view const& target, bool check_bounds,
      rmm::mr::device_memory_resource* mr,
      cudaStream_t stream)
  {
    // TODO
    return std::make_unique<table>(target, stream, mr);
  }

  template <typename map_type, std::enable_if_t<not std::is_integral<map_type>::value
    or std::is_same<map_type, cudf::experimental::bool8>::value>* = nullptr>
  std::unique_ptr<table> operator()(
      table_view const& source, column_view const& scatter_map,
      table_view const& target, bool check_bounds,
      rmm::mr::device_memory_resource* mr,
      cudaStream_t stream)
  {
    CUDF_FAIL("Scatter map column must be an integral, non-boolean type");
  }
};

struct scatter_scalar_impl {
  template <typename map_type, std::enable_if_t<std::is_integral<map_type>::value
    and not std::is_same<map_type, cudf::experimental::bool8>::value>* = nullptr>
  std::unique_ptr<table> operator()(
      std::vector<std::unique_ptr<scalar>> const& source, column_view const& indices,
      table_view const& target, bool check_bounds,
      rmm::mr::device_memory_resource* mr,
      cudaStream_t stream)
  {
    // TODO
    return std::make_unique<table>(target, stream, mr);
  }

  template <typename map_type, std::enable_if_t<not std::is_integral<map_type>::value
    or std::is_same<map_type, cudf::experimental::bool8>::value>* = nullptr>
  std::unique_ptr<table> operator()(
      std::vector<std::unique_ptr<scalar>> const& source, column_view const& indices,
      table_view const& target, bool check_bounds,
      rmm::mr::device_memory_resource* mr,
      cudaStream_t stream)
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
