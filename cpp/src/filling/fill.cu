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
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/mr/device_memory_resource.hpp>

// for gdf_scalar, unnecessary once we switch to cudf::scalar
#include <cudf/types.h>

#include <cuda_runtime.h>

#include <memory>

namespace {

struct scalar_factory {
  gdf_scalar value;

  template <typename T>
  struct scalar {
    T value;
    bool is_valid;

    __device__
    T data(cudf::size_type index) { return value; }

    __device__
    bool valid(cudf::size_type index) { return is_valid; }
  };

  template <typename T>
  scalar<T> make() {
    T val{}; // Safe type pun, compiler should optimize away the memcpy
    memcpy(&val, &value.data, sizeof(T));
    return scalar<T>{val, value.is_valid};
  }
};

}  // namespace

namespace cudf {
namespace experimental {

namespace detail {

void fill(mutable_column_view& destination,
          size_type begin,
          size_type end,
          gdf_scalar const& value,
          cudaStream_t stream) {
  CUDF_EXPECTS(cudf::is_fixed_width(destination.type()) == true,
               "In-place fill does not support variable-sized types.");
  CUDF_EXPECTS((begin >= 0) &&
               (begin <= end) &&
               (begin < destination.size()) &&
               (end <= destination.size()),
               "Range is out of bounds.");
  CUDF_EXPECTS((destination.nullable() == true) || (value.is_valid == true),
               "destination should be nullable or value should be non-null.");
  // The line below should be enabled once we switch from gdf_scalar to
  // cudf::scalar.
  // CUDF_EXPECTS(destination.type() == value.type(), "Data type mismatch.");

  if (end != begin) {  // otherwise no-op
    // The code below is speculative on the future copy_range implementation for
    // cudf::column
    copy_range(destination, scalar_factory{value}, begin, end, stream);
  }

  return;
}

std::unique_ptr<column> fill(column_view const& input,
                             size_type begin,
                             size_type end,
                             gdf_scalar const& value,
                             cudaStream_t stream,
                             rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS(cudf::is_fixed_width(input.type()) == true,
               "Variable-sized types are not supported yet.");
  CUDF_EXPECTS((begin >= 0) &&
               (begin <= end) &&
               (begin < input.size()) &&
               (end <= input.size()),
               "Range is out of bounds.");
  // The line below should be enabled once we switch from gdf_scalar to
  // cudf::scalar.
  // CUDF_EXPECTS(input.type() == value.type(), "Data type mismatch.");

  auto state = mask_state{UNALLOCATED};
  if ((input.nullable() == true) || (value.is_valid == false)) {
    state = UNINITIALIZED;
  }

  auto ret = std::unique_ptr<column>{nullptr};

  if (cudf::is_numeric(input.type()) == true) {
    ret = make_numeric_column(input.type(), input.size(), state, stream, mr);
  }
  else if (cudf::is_timestamp(input.type()) == true) {
    ret = make_timestamp_column(input.type(), input.size(), state, stream, mr);
  }
  else {
    CUDF_FAIL("Unimplemented.");
  }

  auto destination = ret->mutable_view();
  if (begin > 0) {
    copy_range(destination, input, 0, begin, 0, stream);
  }
  if (end != begin) {  // otherwise no fill
    fill(destination, begin, end, value, stream);
  }
  if (end < input.size()) {
    copy_range(destination, input, end, input.size(), end, stream);
  }

  return ret;
}

}  // namespace detail

void fill(mutable_column_view& destination,
          size_type begin,
          size_type end,
          gdf_scalar const& value) {
  return detail::fill(destination, begin, end, value, 0);
}

std::unique_ptr<column> fill(column_view const& input,
                             size_type begin,
                             size_type end,
                             gdf_scalar const& value,
                             rmm::mr::device_memory_resource* mr) {
  return detail::fill(input, begin, end, value, 0, mr);
}

}  // namespace experimental
}  // namespace cudf
