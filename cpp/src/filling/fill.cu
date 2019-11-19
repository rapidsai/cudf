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
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime.h>

#include <memory>

namespace {

struct inplace_fill_range_dispatch {
  cudf::scalar const* p_value = nullptr;
  cudf::mutable_column_view& target;

  template <typename T>
  std::enable_if_t<cudf::is_fixed_width<T>(), void>
  operator()(cudf::size_type source_begin, cudf::size_type source_end,
             cudf::size_type target_begin, cudaStream_t stream = 0) {
    using ScalarType = cudf::experimental::scalar_type_t<T>;
#if 1
    // TODO: temporary till the const issue in cudf::scalar's value() is fixed.
    auto p_scalar =
      const_cast<ScalarType*>(static_cast<ScalarType const*>(this->p_value));
#else
    auto p_scalar = static_cast<ScalarType const*>(this->p_value);
#endif
    T value = p_scalar->value(stream);
    bool is_valid = p_scalar->is_valid();
    cudf::experimental::detail::copy_range(
      thrust::make_constant_iterator(value),
      thrust::make_constant_iterator(is_valid),
      target, target_begin, target_begin + (source_end - source_begin),
      stream);
  }

  template <typename T>
  std::enable_if_t<not cudf::is_fixed_width<T>(), void>
  operator()(cudf::size_type source_begin, cudf::size_type source_end,
             cudf::size_type target_begin, cudaStream_t stream = 0) {
    CUDF_FAIL("in-place fill does not work for variable width types.");
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
      inplace_fill_range_dispatch{&value, destination},
      0, end - begin, begin, stream);
  }

  return;
}

std::unique_ptr<column> fill(column_view const& input,
                             size_type begin,
                             size_type end,
                             scalar const& value,
                             rmm::mr::device_memory_resource* mr,
                             cudaStream_t stream) {
  CUDF_EXPECTS(cudf::is_fixed_width(input.type()) == true,
               "Variable-sized types are not supported yet.");
  CUDF_EXPECTS((begin >= 0) &&
               (begin <= end) &&
               (begin < input.size()) &&
               (end <= input.size()),
               "Range is out of bounds.");
  CUDF_EXPECTS(input.type() == value.type(), "Data type mismatch.");

  auto p_ret = std::make_unique<column>(input, stream, mr);
  if (!p_ret->nullable() && !value.is_valid()) {
    p_ret->set_null_mask(
      create_null_mask(p_ret->size(), ALL_VALID, stream, mr), 0);
  }
  if (end != begin) {  // otherwise no fill
    auto destination = p_ret->mutable_view();
    fill(destination, begin, end, value, stream);
  }

  return p_ret;
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
