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
  cudf::column_view source;
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
  column_range<T> make(cudaStream_t stream = 0) {
    return column_range<T>{
      source.head<T>(),
      source.null_mask(),
      source.offset() + offset,
    };
  }
};

}

namespace cudf {
namespace experimental {

namespace detail {

void copy_range(column_view const& source, mutable_column_view& target,
                size_type source_begin, size_type source_end,
                size_type target_begin,
                cudaStream_t stream) {
  CUDF_EXPECTS(cudf::is_fixed_width(target.type()) == true,
               "In-place copy_range does not support variable-sized types.");
  CUDF_EXPECTS((source_begin <= source_end) &&
                 (source_begin >= 0) &&
                 (source_begin < source.size()) &&
                 (source_end <= source.size()) &&
                 (target_begin >= 0) &&
                 (target_begin < target.size()) &&
                 (target_begin + (source_end - source_begin) <= target.size()),
               "Range is out of bounds.");
  CUDF_EXPECTS(target.type() == source.type(), "Data type mismatch.");
  CUDF_EXPECTS((target.nullable() == true) || (source.has_nulls() == false),
               "target should be nullable if source has null values.");

  if (source_end != source_begin) {  // otherwise no-op
    auto target_end = target_begin + (source_end - source_begin);
    copy_range(
      column_range_factory{source, source_begin},
      target,
      target_begin, target_end, stream);
  }
}

std::unique_ptr<column> copy_range(column_view const& source,
                                   column_view const& target,
                                   size_type source_begin, size_type source_end,
                                   size_type target_begin,
                                   cudaStream_t stream,
                                   rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS(cudf::is_fixed_width(target.type()) == true,
               "Variable-sized types are not supported yet.");
  CUDF_EXPECTS((source_begin >= 0) &&
                 (source_begin <= source_end) &&
                 (source_begin < source.size()) &&
                 (source_end <= source.size()) &&
                 (target_begin >= 0) &&
                 (target_begin < target.size()) &&
                 (target_begin + (source_end - source_begin) <= target.size()),
               "Range is out of bounds.");
  CUDF_EXPECTS(target.type() == source.type(), "Data type mismatch.");

  auto p_ret = std::make_unique<column>(target, stream, mr);
  if ((!p_ret->nullable()) && source.has_nulls()) {
    p_ret->set_null_mask(
      create_null_mask(p_ret->size(), ALL_VALID, stream, mr), 0);
  }
  if (source_end != source_begin) {  // otherwise no-op
    auto ret_view = p_ret->mutable_view();
    copy_range(source, ret_view, source_begin, source_end, target_begin,
               stream);
  }

  return p_ret;
}

}  // namespace detail

void copy_range(column_view const& source, mutable_column_view& target,
                size_type source_begin, size_type source_end,
                size_type target_begin) {
  return detail::copy_range(source, target, source_begin, source_end,
                            target_begin, 0);
}

std::unique_ptr<column> copy_range(column_view const& source,
                                   column_view const& target,
                                   size_type source_begin, size_type source_end,
                                   size_type target_begin,
                                   rmm::mr::device_memory_resource* mr) {
  return detail::copy_range(source, target, source_begin, source_end,
                            target_begin, 0, mr);
}

}  // namespace experimental
} // namespace cudf
