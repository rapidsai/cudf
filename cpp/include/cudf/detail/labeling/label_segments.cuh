/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

namespace cudf::detail {

/**
 * @brief Fill label values for segments defined by a given offsets array.
 *
 * Given a pair of iterators accessing to an offset array, generate label values for segments
 * defined by the offset values. The output will be an array containing consecutive groups of
 * identical labels, the number of elements in each group `i` is defined by
 * `offsets[i+1] - offsets[i]`.
 *
 * The labels always start from `0` regardless of the offset values.
 *
 * @code{.pseudo}
 * Examples:
 *
 * offsets = { 0, 4, 6, 10 }
 * output  = { 0, 0, 0, 0, 1, 1, 2, 2, 2, 2 }
 *
 * offsets = { 5, 10, 12 }
 * output  = { 0, 0, 0, 0, 0, 1, 1 }
 * @endcode
 *
 * @param offsets_begin The beginning of the offsets that define segments.
 * @param offsets_end The end of the offsets that define segments.
 * @param out_begin The beginning of the output label range.
 * @param out_end The end of the output label range.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename InputIterator, typename OutputIterator>
void label_segments(InputIterator offsets_begin,
                    InputIterator offsets_end,
                    OutputIterator out_begin,
                    OutputIterator out_end,
                    rmm::cuda_stream_view stream)
{
  auto const zero_normalized_offsets = thrust::make_transform_iterator(
    offsets_begin, [offsets_begin] __device__(auto const idx) { return idx - *offsets_begin; });

  // The output labels from `upper_bound` will start from `1`.
  // This will shift the result values back to start from `0`.
  using OutputType  = typename thrust::iterator_value<OutputIterator>::type;
  auto const output = thrust::make_transform_output_iterator(
    out_begin, [] __device__(auto const idx) { return idx - OutputType{1}; });

  thrust::upper_bound(rmm::exec_policy(stream),
                      zero_normalized_offsets,
                      zero_normalized_offsets + thrust::distance(offsets_begin, offsets_end),
                      thrust::make_counting_iterator<OutputType>(0),
                      thrust::make_counting_iterator<OutputType>(
                        static_cast<OutputType>(thrust::distance(out_begin, out_end))),
                      output);
}

}  // namespace cudf::detail
