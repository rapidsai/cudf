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

#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace {
struct scalar_construction_helper {
  template <typename T,
            typename ScalarType = scalar_type_t<T>,
            typename std::enable_if_t<is_fixed_width<T>() and not is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<scalar> operator()(cudaStream_t stream, rmm::mr::device_memory_resource* mr) const
  {
    using Type = device_storage_type_t<T>;
    auto s     = new ScalarType(Type{}, false, stream, mr);
    return std::unique_ptr<scalar>(s);
  }

  template <typename T,
            typename ScalarType                             = scalar_type_t<T>,
            typename std::enable_if_t<is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<scalar> operator()(cudaStream_t stream, rmm::mr::device_memory_resource* mr) const
  {
    using Type = device_storage_type_t<T>;
    auto s     = new ScalarType(Type{}, numeric::scale_type{0}, false, stream, mr);
    return std::unique_ptr<scalar>(s);
  }

  template <typename T,
            typename... Args,
            typename std::enable_if_t<not is_fixed_width<T>()>* = nullptr>
  std::unique_ptr<scalar> operator()(Args... args) const
  {
    CUDF_FAIL("Invalid type.");
  }
};
}  // namespace

// Allocate storage for a single numeric element
std::unique_ptr<scalar> make_numeric_scalar(data_type type,
                                            cudaStream_t stream,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(is_numeric(type), "Invalid, non-numeric type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, mr);
}

// Allocate storage for a single timestamp element
std::unique_ptr<scalar> make_timestamp_scalar(data_type type,
                                              cudaStream_t stream,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(is_timestamp(type), "Invalid, non-timestamp type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, mr);
}

// Allocate storage for a single duration element
std::unique_ptr<scalar> make_duration_scalar(data_type type,
                                             cudaStream_t stream,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(is_duration(type), "Invalid, non-duration type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, mr);
}

// Allocate storage for a single fixed width element
std::unique_ptr<scalar> make_fixed_width_scalar(data_type type,
                                                cudaStream_t stream,
                                                rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(is_fixed_width(type), "Invalid, non-fixed-width type.");

  return type_dispatcher(type, scalar_construction_helper{}, stream, mr);
}

namespace {
struct default_scalar_functor {
  template <typename T>
  std::unique_ptr<cudf::scalar> operator()()
  {
    using ScalarType = scalar_type_t<T>;
    return std::unique_ptr<scalar>(new ScalarType);
  }
};

template <>
std::unique_ptr<cudf::scalar> default_scalar_functor::operator()<dictionary32>()
{
  CUDF_FAIL("dictionary type not supported");
}

template <>
std::unique_ptr<cudf::scalar> default_scalar_functor::operator()<list_view>()
{
  CUDF_FAIL("list_view type not supported");
}

template <>
std::unique_ptr<cudf::scalar> default_scalar_functor::operator()<struct_view>()
{
  CUDF_FAIL("struct_view type not supported");
}

}  // namespace

std::unique_ptr<scalar> make_default_constructed_scalar(data_type type)
{
  return type_dispatcher(type, default_scalar_functor{});
}

}  // namespace cudf
