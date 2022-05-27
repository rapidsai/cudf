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
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/distance.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
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
 * Note that the caller is responsible to make sure the output range have the correct size, which is
 * the total segment sizes (i.e., `size = *(offsets_end - 1) - *offsets_begin`). Otherwise, the
 * result is undefined.
 *
 * @code{.pseudo}
 * Examples:
 *
 * offsets = [ 0, 4, 6, 6, 6, 10 ]
 * output  = [ 0, 0, 0, 0, 1, 1, 4, 4, 4, 4 ]
 *
 * offsets = [ 5, 10, 12 ]
 * output  = [ 0, 0, 0, 0, 0, 1, 1 ]
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
                     // Zero-normalized offsets.
                     auto const dst_idx = offsets[idx] - offsets[0];

                     // Scatter value `1` to the index at offsets[idx].
                     // In case we have repeated offsets (i.e., we have empty segments), this
                     // atomicAdd call will make sure the label values corresponding to these empty
                     // segments will be skipped in the output.
                     atomicAdd(&output[dst_idx], OutputType{1});
                   });
  thrust::inclusive_scan(rmm::exec_policy(stream), out_begin, out_end, out_begin);
}

/**
 * @brief Generate segment offsets from groups of identical label values.
 *
 * Given a pair of iterators accessing to an array containing groups of identical label values,
 * generate offsets for segments defined by these label.
 *
 * Empty segments are also taken into account. If the input label values are discontinuous, the
 * segments corresponding to the missing labels will be inferred as empty segments and their offsets
 * will also be generated.
 *
 * Note that the caller is responsible to make sure the output range for offsets have the correct
 * size, which is the maximum label value plus two (i.e., `size = *(labels_end - 1) + 2`).
 * Otherwise, the result is undefined.
 *
 * @code{.pseudo}
 * Examples:
 *
 * labels = [ 0, 0, 0, 0, 1, 1, 4, 4, 4, 4 ]
 * output = [ 0, 4, 6, 6, 6, 10 ]
 *
 * labels = [ 0, 0, 0, 0, 0, 1, 1 ]
 * output = [ 0, 5, 7 ]
 * @endcode
 *
 * @param labels_begin The beginning of the labels that define segments.
 * @param labels_end The end of the labels that define segments.
 * @param out_begin The beginning of the output offset range.
 * @param out_end The end of the output offset range.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename InputIterator, typename OutputIterator>
void labels_to_offsets(InputIterator labels_begin,
                       InputIterator labels_end,
                       OutputIterator out_begin,
                       OutputIterator out_end,
                       rmm::cuda_stream_view stream)
{
  // The output offsets need to be filled with `0` value first.
  using OutputType = typename thrust::iterator_value<OutputIterator>::type;
  thrust::uninitialized_fill(rmm::exec_policy(stream), out_begin, out_end, OutputType{0});

  auto const num_labels = static_cast<size_type>(thrust::distance(labels_begin, labels_end));
  if (num_labels == 0) { return; }

  auto const num_segments = static_cast<size_type>(thrust::distance(out_begin, out_end)) - 1;

  //================================================================================
  // Let consider an example: Given input labels = [ 0, 0, 0, 0, 1, 1, 4, 4, 4, 4 ].

  // This stores the unique label values.
  // Given the example above, we will have this array containing [0, 1, 4].
  auto list_indices = rmm::device_uvector<size_type>(num_segments, stream);

  // Stores the non-zero segment sizes.
  // Given the example above, we will have this array containing [4, 2, 4].
  auto list_sizes = rmm::device_uvector<size_type>(num_segments, stream);

  // Count the numbers of unique labels in the input.
  auto const end                    = thrust::reduce_by_key(rmm::exec_policy(stream),
                                         labels_begin,  // keys
                                         labels_end,    // keys
                                         thrust::make_constant_iterator<size_type>(1),
                                         list_indices.begin(),  // output unique input labels
                                         list_sizes.begin());  // count for each label
  auto const num_non_empty_segments = thrust::distance(list_indices.begin(), end.first);

  // Scatter segment sizes into the end position of their corresponding segment indices.
  // Given the example above, we scatter [4, 2, 4] by the scatter_map [0, 1, 4], resulting
  // output = [4, 2, 0, 0, 4, 0].
  thrust::scatter(rmm::exec_policy(stream),
                  list_sizes.begin(),
                  list_sizes.begin() + num_non_empty_segments,
                  list_indices.begin(),
                  out_begin);

  // Generate offsets from sizes.
  // Given the example above, the final output is [0, 4, 6, 6, 6, 10].
  thrust::exclusive_scan(rmm::exec_policy(stream), out_begin, out_end, out_begin);
}

}  // namespace cudf::detail
