/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#pragma once

#include "reduction_operators.cuh"

#include <cudf/detail/utilities/cast_functor.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/transform.h>

namespace cudf {
namespace reduction {
namespace detail {

/**
 * @brief Compute the specified simple reduction over each of the segments in the
 * input range of elements
 *
 * @tparam InputIterator    Input iterator type
 * @tparam OffsetIterator   Offset iterator type
 * @tparam OutputIterator   Output iterator type
 * @tparam BinaryOp         Binary operator used for reduce
 * @tparam OutputType       The output type derived from the OutputIterator
 *
 * @param d_in           Input data iterator
 * @param d_offset_begin Begin iterator to segment indices
 * @param d_offset_end   End iterator to segment indices
 * @param d_out          Output data iterator
 * @param op             The reduction operator
 * @param initial_value  Initial value of the reduction
 * @param stream         CUDA stream used for device memory operations and kernel launches
 *
 */
template <typename InputIterator,
          typename OffsetIterator,
          typename OutputIterator,
          typename BinaryOp,
          typename OutputType = typename thrust::iterator_value<OutputIterator>::type,
          typename std::enable_if_t<is_fixed_width<OutputType>() &&
                                    !cudf::is_fixed_point<OutputType>()>* = nullptr>
void segmented_reduce(InputIterator d_in,
                      OffsetIterator d_offset_begin,
                      OffsetIterator d_offset_end,
                      OutputIterator d_out,
                      BinaryOp op,
                      OutputType initial_value,
                      rmm::cuda_stream_view stream)
{
  auto const num_segments = static_cast<size_type>(std::distance(d_offset_begin, d_offset_end)) - 1;
  auto const binary_op    = cudf::detail::cast_functor<OutputType>(op);
  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(nullptr,
                                     temp_storage_bytes,
                                     d_in,
                                     d_out,
                                     num_segments,
                                     d_offset_begin,
                                     d_offset_begin + 1,
                                     binary_op,
                                     initial_value,
                                     stream.value());
  auto d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};

  // Run reduction
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage.data(),
                                     temp_storage_bytes,
                                     d_in,
                                     d_out,
                                     num_segments,
                                     d_offset_begin,
                                     d_offset_begin + 1,
                                     binary_op,
                                     initial_value,
                                     stream.value());
}

template <typename InputIterator,
          typename OffsetIterator,
          typename OutputIterator,
          typename BinaryOp,
          typename OutputType = typename thrust::iterator_value<OutputIterator>::type,
          typename std::enable_if_t<!(is_fixed_width<OutputType>() &&
                                      !cudf::is_fixed_point<OutputType>())>* = nullptr>
void segmented_reduce(InputIterator,
                      OffsetIterator,
                      OffsetIterator,
                      OutputIterator,
                      BinaryOp,
                      OutputType,
                      rmm::cuda_stream_view)
{
  CUDF_FAIL(
    "Unsupported data types called on segmented_reduce. Only numeric and chrono types are "
    "supported.");
}

/**
 * @brief Compute reduction by the compound operator (reduce and transform)
 *
 * The reduction operator must have an `intermediate::compute_result()` method.
 * This method performs reduction using binary operator `Op::Op` and calculates the
 * result to `OutputType` using `compute_result()` through the transform method.
 *
 * @tparam Op              Reduction operator
 * @tparam InputIterator   Input iterator type
 * @tparam OffsetIterator  Offsets iterator type
 * @tparam OutputIterator  Output iterator type
 *
 * @param d_in           Input data iterator
 * @param d_offset_begin Begin iterator to segment indices
 * @param d_offset_end   End iterator to segment indices
 * @param d_out          Output data iterator
 * @param op             The reduction operator
 * @param ddof           Delta degrees of freedom used for standard deviation and variance
 * @param d_valid_counts Number of valid values per segment
 * @param stream         CUDA stream used for device memory operations and kernel launches
 */
template <typename Op, typename InputIterator, typename OffsetIterator, typename OutputIterator>
void segmented_reduce(InputIterator d_in,
                      OffsetIterator d_offset_begin,
                      OffsetIterator d_offset_end,
                      OutputIterator d_out,
                      op::compound_op<Op> op,
                      size_type ddof,
                      size_type* d_valid_counts,
                      rmm::cuda_stream_view stream)
{
  using OutputType       = typename thrust::iterator_value<OutputIterator>::type;
  using IntermediateType = typename thrust::iterator_value<InputIterator>::type;
  auto num_segments      = static_cast<size_type>(std::distance(d_offset_begin, d_offset_end)) - 1;
  auto const initial_value = op.template get_identity<IntermediateType>();
  auto const binary_op     = cudf::detail::cast_functor<IntermediateType>(op.get_binary_op());

  rmm::device_uvector<IntermediateType> intermediate_result{static_cast<std::size_t>(num_segments),
                                                            stream};

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(nullptr,
                                     temp_storage_bytes,
                                     d_in,
                                     intermediate_result.data(),
                                     num_segments,
                                     d_offset_begin,
                                     d_offset_begin + 1,
                                     binary_op,
                                     initial_value,
                                     stream.value());
  auto d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};

  // Run reduction
  cub::DeviceSegmentedReduce::Reduce(d_temp_storage.data(),
                                     temp_storage_bytes,
                                     d_in,
                                     intermediate_result.data(),
                                     num_segments,
                                     d_offset_begin,
                                     d_offset_begin + 1,
                                     binary_op,
                                     initial_value,
                                     stream.value());

  // compute the result value from intermediate value in device
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(num_segments),
    d_out,
    [ir = intermediate_result.data(), op, d_valid_counts, ddof] __device__(auto idx) {
      auto const count = d_valid_counts[idx];
      return count > 0 ? op.template compute_result<OutputType>(ir[idx], count, ddof)
                       : OutputType{0};
    });
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
