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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/repeat.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scan.h>

#include <thrust/copy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

#include <limits>
#include <memory>

namespace cudf {
namespace detail {
rmm::device_uvector<size_type> complement(device_span<size_type const> const& input,
                                          size_type size,
                                          out_of_bounds_check bounds_check,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(size >= 0, "size value must be non-negative.");

  // An temporary array to mark if a value `i` (at index `i`) exists in the input.
  // This array may have one extra value at the end for discarding the out-of-bounds input.
  auto labels =
    rmm::device_uvector<int8_t>(size + (bounds_check == out_of_bounds_check::YES), stream);

  auto constexpr EXIST_LABEL    = std::numeric_limits<int8_t>::max();
  auto constexpr NONEXIST_LABEL = std::numeric_limits<int8_t>::min();

  // Firstly, mark all values as "non-exist". Then, for each input value, mark it as "exist".
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), labels.begin(), labels.end(), NONEXIST_LABEL);

  auto const do_scatter = [&](auto const& input_begin) {
    thrust::scatter(rmm::exec_policy(stream),
                    thrust::make_constant_iterator(EXIST_LABEL),
                    thrust::make_constant_iterator(EXIST_LABEL) + input.size(),
                    input_begin,
                    labels.begin());
  };

  if (bounds_check == out_of_bounds_check::YES) {
    // If any input value is out-of-bounds (i.e., out side of [0, size)), just scatter it to the
    // last position in the labels array.
    // This position will be then ignored from consideration for the output.
    auto const normalized_input_begin = thrust::make_transform_iterator(
      input.begin(), [size] __device__(auto val) { return (val < 0 || val >= size) ? size : val; });
    do_scatter(normalized_input_begin);
  } else {
    do_scatter(input.begin());
  }

  // Finally, copy the values that are still marked as "non-exist" to the output.
  auto result = rmm::device_uvector<size_type>(size, stream, mr);
  auto const copy_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(size),
                    labels.begin(),
                    result.begin(),
                    [NONEXIST_LABEL] __device__(auto const val) { return val == NONEXIST_LABEL; });

  result.resize(thrust::distance(result.begin(), copy_end), stream);
  return result;
}

}  // namespace detail

rmm::device_uvector<size_type> complement(device_span<size_type const> const& input,
                                          size_type size,
                                          out_of_bounds_check bounds_check,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::complement(input, size, bounds_check, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
