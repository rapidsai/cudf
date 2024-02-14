/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

namespace cudf::detail {

template <typename Iterator, typename T, typename BinaryOp>
__device__ __forceinline__ T accumulate(Iterator first, Iterator last, T init, BinaryOp op)
{
  for (; first != last; ++first) {
    init = op(std::move(init), *first);
  }
  return init;
}

/**
 * @copydoc cudf::detail::copy_if_safe(rmm::exec_policy, InputIterator, InputIterator,
 * OutputIterator, Predicate, rmm::cuda_stream_view)
 *
 * @tparam StencilIterator Type of the stencil iterator
 * @param stencil The beginning of the stencil sequence
 */
template <typename InputIterator,
          typename StencilIterator,
          typename OutputIterator,
          typename Predicate>
OutputIterator copy_if_safe(InputIterator first,
                            InputIterator last,
                            StencilIterator stencil,
                            OutputIterator result,
                            Predicate pred,
                            rmm::cuda_stream_view stream)
{
  auto const copy_size = std::min(static_cast<std::size_t>(std::distance(first, last)),
                                  static_cast<std::size_t>(std::numeric_limits<int>::max()));

  auto itr = first;
  while (itr != last) {
    auto const copy_end =
      static_cast<std::size_t>(std::distance(itr, last)) <= copy_size ? last : itr + copy_size;
    result = thrust::copy_if(rmm::exec_policy(stream), itr, copy_end, stencil, result, pred);
    stencil += std::distance(itr, copy_end);
    itr = copy_end;
  }
  return result;
}

/**
 * @brief Utility for calling `thrust::copy_if`.
 *
 * This is a proxy for `thrust::copy_if` which is a workaround for its bug
 * (https://github.com/NVIDIA/thrust/issues/1302) where it cannot iterate over int-max values
 * `distance(first,last) > int-max` This calls thrust::copy_if in 2B chunks instead.
 *
 * @tparam InputIterator Type of the input iterator
 * @tparam OutputIterator Type of the output iterator
 * @tparam Predicate Type of the binary predicate used to determine elements to copy
 *
 * @param first The beginning of the sequence from which to copy
 * @param last The end of the sequence from which to copy
 * @param result The beginning of the sequence into which to copy
 * @param pred The predicate to test on every value of the range `[first, last)`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return An iterator pointing to the position `result + n`, where `n` is equal to the number of
 *         times `pred` evaluated to `true` in the range `[first, last)`.
 */
template <typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator copy_if_safe(InputIterator first,
                            InputIterator last,
                            OutputIterator result,
                            Predicate pred,
                            rmm::cuda_stream_view stream)
{
  auto const copy_size = std::min(static_cast<std::size_t>(std::distance(first, last)),
                                  static_cast<std::size_t>(std::numeric_limits<int>::max()));

  auto itr = first;
  while (itr != last) {
    auto const copy_end =
      static_cast<std::size_t>(std::distance(itr, last)) <= copy_size ? last : itr + copy_size;
    result = thrust::copy_if(rmm::exec_policy(stream), itr, copy_end, result, pred);
    itr    = copy_end;
  }
  return result;
}

}  // namespace cudf::detail
