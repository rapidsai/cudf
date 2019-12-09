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

#include <cudf/filling.hpp>
#include <cudf/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy_range.cuh>
#include <cudf/detail/fill.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/fill.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime.h>

#include <memory>

namespace {

template <typename T>
void in_place_fill(cudf::mutable_column_view& destination,
                   cudf::size_type begin,
                   cudf::size_type end,
                   cudf::scalar const& value,
                   cudaStream_t stream = 0) {
    using ScalarType = cudf::experimental::scalar_type_t<T>;
    auto p_scalar = static_cast<ScalarType const*>(&value);
    T fill_value = p_scalar->value(stream);
    bool is_valid = p_scalar->is_valid();
    cudf::experimental::detail::copy_range(
      thrust::make_constant_iterator(fill_value),
      thrust::make_constant_iterator(is_valid),
      destination, begin, end, stream);
}

struct in_place_fill_range_dispatch {
  cudf::scalar const& value;
  cudf::mutable_column_view& destination;

  template <typename T>
  std::enable_if_t<cudf::is_fixed_width<T>(), void>
  operator()(cudf::size_type begin, cudf::size_type end,
             cudaStream_t stream = 0) {
    in_place_fill<T>(destination, begin, end, value, stream);
  }

  template <typename T>
  std::enable_if_t<not cudf::is_fixed_width<T>(), void>
  operator()(cudf::size_type begin, cudf::size_type end,
             cudaStream_t stream = 0) {
    CUDF_FAIL("in-place fill does not work for variable width types.");
  }
};

struct out_of_place_fill_range_dispatch {
  cudf::scalar const& value;
  cudf::column_view const& input;

  template <typename T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<cudf::column>>
  operator()(
      cudf::size_type begin, cudf::size_type end,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(), 
      cudaStream_t stream = 0) {
    auto p_ret = std::make_unique<cudf::column>(input, stream, mr);

    if (end != begin) {  // otherwise no fill
      if (!p_ret->nullable() && !value.is_valid()) {
        p_ret->set_null_mask(
          cudf::create_null_mask(p_ret->size(), cudf::ALL_VALID, stream, mr), 0);
      }

      auto ret_view = p_ret->mutable_view();
      in_place_fill<T>(ret_view, begin, end, value, stream);
    }

    return p_ret;
  }

  template <typename T>
  std::enable_if_t<std::is_same<cudf::string_view, T>::value,
                   std::unique_ptr<cudf::column>>
  operator()(
      cudf::size_type begin, cudf::size_type end,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(), 
      cudaStream_t stream = 0) {
    using ScalarType = cudf::experimental::scalar_type_t<T>;
    auto p_scalar = static_cast<ScalarType const*>(&value);
    return cudf::strings::detail::fill(cudf::strings_column_view(input),
                                       begin, end, *p_scalar, mr, stream);
  }
};

}  // namespace

namespace cudf {
namespace experimental {

namespace detail {

void fill(mutable_column_view& destination,
          size_type begin,
          size_type end,
          scalar const& value,
          cudaStream_t stream) {
  CUDF_EXPECTS(cudf::is_fixed_width(destination.type()) == true,
               "In-place fill does not support variable-sized types.");
  CUDF_EXPECTS((begin >= 0) &&
               (begin <= end) &&
               (begin < destination.size()) &&
               (end <= destination.size()),
               "Range is out of bounds.");
  CUDF_EXPECTS((destination.nullable() == true) || (value.is_valid() == true),
               "destination should be nullable or value should be non-null.");
  CUDF_EXPECTS(destination.type() == value.type(), "Data type mismatch.");

  if (end != begin) {  // otherwise no-op
    cudf::experimental::type_dispatcher(
      destination.type(),
      in_place_fill_range_dispatch{value, destination},
      begin, end, stream);
  }

  return;
}

std::unique_ptr<column> fill(column_view const& input,
                             size_type begin,
                             size_type end,
                             scalar const& value,
                             rmm::mr::device_memory_resource* mr,
                             cudaStream_t stream) {
  CUDF_EXPECTS((begin >= 0) &&
               (begin <= end) &&
               (begin < input.size()) &&
               (end <= input.size()),
               "Range is out of bounds.");
  CUDF_EXPECTS(input.type() == value.type(), "Data type mismatch.");

  return cudf::experimental::type_dispatcher(
      input.type(),
      out_of_place_fill_range_dispatch{value, input},
      begin, end, mr, stream);
}

}  // namespace detail

void fill(mutable_column_view& destination,
          size_type begin,
          size_type end,
          scalar const& value) {
  return detail::fill(destination, begin, end, value, 0);
}

std::unique_ptr<column> fill(column_view const& input,
                             size_type begin,
                             size_type end,
                             scalar const& value,
                             rmm::mr::device_memory_resource* mr) {
  return detail::fill(input, begin, end, value, mr, 0);
}

}  // namespace experimental
}  // namespace cudf
