/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/copy.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace {
struct scalar_construction_helper {
  template <typename T,
            typename ScalarType                                                = scalar_type_t<T>,
            std::enable_if_t<is_fixed_width<T>() and not is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<scalar> operator()(rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    using Type = device_storage_type_t<T>;
    auto s     = new ScalarType(Type{}, false, stream, mr);
    return std::unique_ptr<scalar>(s);
  }

  template <typename T,
            typename ScalarType                    = scalar_type_t<T>,
            std::enable_if_t<is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<scalar> operator()(rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    using Type = device_storage_type_t<T>;
    auto s     = new ScalarType(Type{}, numeric::scale_type{0}, false, stream, mr);
    return std::unique_ptr<scalar>(s);
  }

  template <typename T, typename... Args, std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<scalar> operator()(Args... args) const
  {
    CUDF_FAIL("Invalid type.");
  }
};
}  // namespace

// Allocate storage for a single numeric element
std::unique_ptr<scalar> make_numeric_scalar(data_type type,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(is_numeric(type), "Invalid, non-numeric type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, mr);
}

// Allocate storage for a single timestamp element
std::unique_ptr<scalar> make_timestamp_scalar(data_type type,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(is_timestamp(type), "Invalid, non-timestamp type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, mr);
}

// Allocate storage for a single duration element
std::unique_ptr<scalar> make_duration_scalar(data_type type,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(is_duration(type), "Invalid, non-duration type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, mr);
}

// Allocate storage for a single fixed width element
std::unique_ptr<scalar> make_fixed_width_scalar(data_type type,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(is_fixed_width(type), "Invalid, non-fixed-width type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, mr);
}

std::unique_ptr<scalar> make_list_scalar(column_view elements,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  return std::make_unique<list_scalar>(elements, true, stream, mr);
}

std::unique_ptr<scalar> make_struct_scalar(table_view const& data,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  return std::make_unique<struct_scalar>(data, true, stream, mr);
}

std::unique_ptr<scalar> make_struct_scalar(host_span<column_view const> data,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  return std::make_unique<struct_scalar>(data, true, stream, mr);
}

namespace {
struct default_scalar_functor {
  data_type type;

  template <typename T, std::enable_if_t<not is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<cudf::scalar> operator()(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    return make_fixed_width_scalar(data_type(type_to_id<T>()), stream, mr);
  }

  template <typename T, std::enable_if_t<is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<cudf::scalar> operator()(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    auto const scale_ = numeric::scale_type{type.scale()};
    auto s            = make_fixed_point_scalar<T>(0, scale_, stream, mr);
    s->set_valid_async(false, stream);
    return s;
  }
};

template <>
std::unique_ptr<cudf::scalar> default_scalar_functor::operator()<string_view>(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  return std::unique_ptr<scalar>(new string_scalar("", false, stream, mr));
}

template <>
std::unique_ptr<cudf::scalar> default_scalar_functor::operator()<dictionary32>(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  CUDF_FAIL("dictionary type not supported");
}

template <>
std::unique_ptr<cudf::scalar> default_scalar_functor::operator()<list_view>(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  CUDF_FAIL("list_view type not supported");
}

template <>
std::unique_ptr<cudf::scalar> default_scalar_functor::operator()<struct_view>(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  CUDF_FAIL("struct_view type not supported");
}

}  // namespace

std::unique_ptr<scalar> make_default_constructed_scalar(data_type type,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  return type_dispatcher(type, default_scalar_functor{type}, stream, mr);
}

std::unique_ptr<scalar> make_empty_scalar_like(column_view const& column,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  std::unique_ptr<scalar> result;
  switch (column.type().id()) {
    case type_id::LIST: {
      auto const empty_child = empty_like(lists_column_view(column).child());
      result                 = make_list_scalar(empty_child->view(), stream, mr);
      result->set_valid_async(false, stream);
      break;
    }
    case type_id::STRUCT:
      // The input column must have at least 1 row to extract a scalar (row) from it.
      result = detail::get_element(column, 0, stream, mr);
      result->set_valid_async(false, stream);
      break;
    default: result = make_default_constructed_scalar(column.type(), stream, mr);
  }
  return result;
}

}  // namespace cudf
