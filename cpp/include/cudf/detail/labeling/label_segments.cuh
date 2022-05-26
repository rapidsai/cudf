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

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/uninitialized_fill.h>

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
 * In case there are empty segments, their corresponding label values will be skipped in the output.
 *
 * @code{.pseudo}
 * Examples:
 *
 * offsets = { 0, 4, 6, 6, 6, 10 }
 * output  = { 0, 0, 0, 0, 1, 1, 4, 4, 4, 4 }
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
  auto const num_segments =
    static_cast<size_type>(thrust::distance(offsets_begin, offsets_end)) - 1;
  if (num_segments <= 0) { return; }

  using OutputType = typename thrust::iterator_value<OutputIterator>::type;
  thrust::uninitialized_fill(rmm::exec_policy(stream), out_begin, out_end, OutputType{0});
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(size_type{1}),
                   thrust::make_counting_iterator(num_segments),
                   [offsets = offsets_begin, output = out_begin] __device__(auto const idx) {
                     auto const dst_idx = offsets[idx] - offsets[0];

                     // Scatter value `1` to the index at offsets[idx].
                     // In case we have repeated offsets (i.e., we have empty segments), this
                     // atomicAdd call will make sure the label values corresponding to these empty
                     // segments will be skipped in the output.
                     atomicAdd(&output[dst_idx], OutputType{1});
                   });
  thrust::inclusive_scan(rmm::exec_policy(stream), out_begin, out_end, out_begin);
}

}  // namespace cudf::detail
