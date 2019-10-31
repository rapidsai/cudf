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

#include <utilities/bit_util.cuh>

#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy_range.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime.h>

#include <memory>

namespace {

struct column_range_factory {
  cudf::column_view input;
  cudf::size_type offset;

  template <typename T>
  struct column_range {
    T const* column_data;
    cudf::bitmask_type const* mask;
    cudf::size_type begin;

    __device__
    T data(cudf::size_type index) { 
      return column_data[begin + index]; }

    __device__
    bool valid(cudf::size_type index) {
      return cudf::util::bit_is_set(mask, begin + index);
    }
  };

  template <typename T>
  column_range<T> make() {
    return column_range<T>{
      input.head<T>(),
      input.null_mask(),
      input.offset() + offset,
    };
  }
};

}

namespace cudf {
namespace experimental {

namespace detail {

void copy_range(mutable_column_view& output, column_view const& input,
                size_type out_begin, size_type out_end, size_type in_begin,
                cudaStream_t stream) {
  CUDF_EXPECTS(cudf::is_fixed_width(output.type()) == true,
               "In-place copy_range does not support variable-sized types.");
  CUDF_EXPECTS((out_begin >= 0) &&
               (out_begin <= out_end) &&
               (out_begin < output.size()) &&
               (out_end <= output.size()) &&
               (in_begin >= 0) &&
               (in_begin < input.size()) &&
               (in_begin + out_end - out_begin <= input.size()),
               "Range is out of bounds.");
  CUDF_EXPECTS(output.type() == input.type(), "Data type mismatch.");
  CUDF_EXPECTS((output.nullable() == true) || (input.has_nulls() == false),
               "output should be nullable if input has null values.");

  if (out_end != out_begin) {  // otherwise no-op
    copy_range(
      output,
      column_range_factory{input, in_begin},
      out_begin, out_end, stream);
  }

}

std::unique_ptr<column> copy_range(column_view const& output,
                                   column_view const& input,
                                   size_type out_begin, size_type out_end,
                                   size_type in_begin,
                                   cudaStream_t stream,
                                   rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS(cudf::is_fixed_width(output.type()) == true,
               "Variable-sized types are not supported yet.");
  CUDF_EXPECTS((out_begin >= 0) &&
               (out_begin <= out_end) &&
               (out_begin < output.size()) &&
               (out_end <= output.size()) &&
               (in_begin >= 0) &&
               (in_begin < input.size()) &&
               (in_begin + out_end - out_begin <= input.size()),
               "Range is out of bounds.");
  CUDF_EXPECTS(output.type() == input.type(), "Data type mismatch.");

  auto state = mask_state{UNALLOCATED};
  if (input.has_nulls() == true) {
    state = UNINITIALIZED;
  }

  auto ret = std::unique_ptr<column>{nullptr};

  if (cudf::is_numeric(output.type()) == true) {
    ret = make_numeric_column(output.type(), output.size(), state, stream, mr);
  }
  else if (cudf::is_timestamp(output.type()) == true) {
    ret = make_timestamp_column(output.type(), output.size(), state, stream, mr);
  }
  else {
    CUDF_FAIL("Unimplemented.");
  }

  auto ret_view = ret->mutable_view();
  if (out_begin > 0) {
    copy_range(ret_view, output, 0, out_begin, 0, stream);
  }
  if (out_end != out_begin) {
    copy_range(ret_view, input, out_begin, out_end, in_begin, stream);
  }
  if (out_end < output.size()) {
    copy_range(ret_view, output, out_end, output.size(), out_end, stream);
  }

  return ret;
}

}  // namespace detail

void copy_range(mutable_column_view& output, column_view const& input,
                size_type out_begin, size_type out_end, size_type in_begin) {
  return detail::copy_range(output, input, out_begin, out_end, in_begin, 0);
}

std::unique_ptr<column> copy_range(column_view const& output,
                                   column_view const& input,
                                   size_type out_begin, size_type out_end,
                                   size_type in_begin,
                                   rmm::mr::device_memory_resource* mr) {
  return detail::copy_range(output, input, out_begin, out_end, in_begin, 0, mr);
}

}  // namespace experimental
} // namespace cudf
