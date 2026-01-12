/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
            std::enable_if_t<is_fixed_width<T>() and not is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<scalar> operator()(rmm::cuda_stream_view stream,
                                     cudf::memory_resources resources) const
  {
    using Type       = device_storage_type_t<T>;
    using ScalarType = scalar_type_t<T>;
    return std::make_unique<ScalarType>(Type{}, false, stream, resources);
  }

  template <typename T, std::enable_if_t<is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<scalar> operator()(rmm::cuda_stream_view stream,
                                     cudf::memory_resources resources) const
  {
    using Type       = device_storage_type_t<T>;
    using ScalarType = scalar_type_t<T>;
    return std::make_unique<ScalarType>(Type{}, numeric::scale_type{0}, false, stream, resources);
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
                                            cudf::memory_resources resources)
{
  CUDF_EXPECTS(is_numeric(type), "Invalid, non-numeric type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, resources);
}

// Allocate storage for a single timestamp element
std::unique_ptr<scalar> make_timestamp_scalar(data_type type,
                                              rmm::cuda_stream_view stream,
                                              cudf::memory_resources resources)
{
  CUDF_EXPECTS(is_timestamp(type), "Invalid, non-timestamp type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, resources);
}

// Allocate storage for a single duration element
std::unique_ptr<scalar> make_duration_scalar(data_type type,
                                             rmm::cuda_stream_view stream,
                                             cudf::memory_resources resources)
{
  CUDF_EXPECTS(is_duration(type), "Invalid, non-duration type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, resources);
}

// Allocate storage for a single fixed width element
std::unique_ptr<scalar> make_fixed_width_scalar(data_type type,
                                                rmm::cuda_stream_view stream,
                                                cudf::memory_resources resources)
{
  CUDF_EXPECTS(is_fixed_width(type), "Invalid, non-fixed-width type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, resources);
}

std::unique_ptr<scalar> make_list_scalar(column_view elements,
                                         rmm::cuda_stream_view stream,
                                         cudf::memory_resources resources)
{
  return std::make_unique<list_scalar>(elements, true, stream, resources);
}

std::unique_ptr<scalar> make_struct_scalar(table_view const& data,
                                           rmm::cuda_stream_view stream,
                                           cudf::memory_resources resources)
{
  return std::make_unique<struct_scalar>(data, true, stream, resources);
}

std::unique_ptr<scalar> make_struct_scalar(host_span<column_view const> data,
                                           rmm::cuda_stream_view stream,
                                           cudf::memory_resources resources)
{
  return std::make_unique<struct_scalar>(data, true, stream, resources);
}

namespace {
struct default_scalar_functor {
  data_type type;

  template <typename T, std::enable_if_t<not is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<cudf::scalar> operator()(rmm::cuda_stream_view stream,
                                           cudf::memory_resources resources)
  {
    return make_fixed_width_scalar(data_type(type_to_id<T>()), stream, resources);
  }

  template <typename T, std::enable_if_t<is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<cudf::scalar> operator()(rmm::cuda_stream_view stream,
                                           cudf::memory_resources resources)
  {
    auto const scale_ = numeric::scale_type{type.scale()};
    auto s            = make_fixed_point_scalar<T>(0, scale_, stream, resources);
    s->set_valid_async(false, stream);
    return s;
  }
};

template <>
std::unique_ptr<cudf::scalar> default_scalar_functor::operator()<string_view>(
  rmm::cuda_stream_view stream, cudf::memory_resources resources)
{
  return std::unique_ptr<scalar>(new string_scalar("", false, stream, resources));
}

template <>
std::unique_ptr<cudf::scalar> default_scalar_functor::operator()<dictionary32>(
  rmm::cuda_stream_view stream, cudf::memory_resources resources)
{
  CUDF_FAIL("dictionary type not supported");
}

template <>
std::unique_ptr<cudf::scalar> default_scalar_functor::operator()<list_view>(
  rmm::cuda_stream_view stream, cudf::memory_resources resources)
{
  CUDF_FAIL("list_view type not supported");
}

template <>
std::unique_ptr<cudf::scalar> default_scalar_functor::operator()<struct_view>(
  rmm::cuda_stream_view stream, cudf::memory_resources resources)
{
  CUDF_FAIL("struct_view type not supported");
}

}  // namespace

std::unique_ptr<scalar> make_default_constructed_scalar(data_type type,
                                                        rmm::cuda_stream_view stream,
                                                        cudf::memory_resources resources)
{
  return type_dispatcher(type, default_scalar_functor{type}, stream, resources);
}

std::unique_ptr<scalar> make_empty_scalar_like(column_view const& column,
                                               rmm::cuda_stream_view stream,
                                               cudf::memory_resources resources)
{
  std::unique_ptr<scalar> result;
  switch (column.type().id()) {
    case type_id::LIST: {
      auto const empty_child = empty_like(lists_column_view(column).child());
      result                 = make_list_scalar(empty_child->view(), stream, resources);
      result->set_valid_async(false, stream);
      break;
    }
    case type_id::STRUCT:
      // The input column must have at least 1 row to extract a scalar (row) from it.
      result = detail::get_element(column, 0, stream, resources);
      result->set_valid_async(false, stream);
      break;
    default: result = make_default_constructed_scalar(column.type(), stream, resources);
  }
  return result;
}

}  // namespace cudf
