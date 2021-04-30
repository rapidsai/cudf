/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include "range_window_bounds_detail.hpp"

namespace cudf {
namespace {

/**
 * @brief Factory to (copy) construct scalars.
 *
 * Derived types of scalars are cloned, to be adopted by `range_window_bounds`.
 * This makes it possible to copy construct and copy assign `range_window_bounds` objects.
 */
struct range_scalar_constructor {
  template <typename T, std::enable_if_t<!detail::is_supported_range_type<T>(), void>* = nullptr>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar_) const
  {
    CUDF_FAIL(
      "Unsupported range type. "
      "Only Durations and non-boolean integral range types are allowed.");
  }

  template <typename T, std::enable_if_t<cudf::is_duration<T>(), void>* = nullptr>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar_) const
  {
    return std::make_unique<duration_scalar<T>>(
      static_cast<duration_scalar<T> const&>(range_scalar_));
  }

  template <typename T,
            std::enable_if_t<std::is_integral<T>::value && !cudf::is_boolean<T>(), void>* = nullptr>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar_) const
  {
    return std::make_unique<numeric_scalar<T>>(
      static_cast<numeric_scalar<T> const&>(range_scalar_));
  }
};

}  // namespace

range_window_bounds::range_window_bounds(bool is_unbounded_, std::unique_ptr<scalar> range_scalar_)
  : _is_unbounded{is_unbounded_}, _range_scalar{std::move(range_scalar_)}
{
  CUDF_EXPECTS(_range_scalar.get(), "Range window scalar cannot be null.");
  CUDF_EXPECTS(_is_unbounded || _range_scalar->is_valid(),
               "Bounded Range window scalar must be valid.");
}

range_window_bounds range_window_bounds::unbounded(data_type type)
{
  return range_window_bounds(true, make_default_constructed_scalar(type));
}

range_window_bounds range_window_bounds::get(scalar const& scalar_)
{
  return range_window_bounds{
    false, cudf::type_dispatcher(scalar_.type(), range_scalar_constructor{}, scalar_)};
}

}  // namespace cudf
