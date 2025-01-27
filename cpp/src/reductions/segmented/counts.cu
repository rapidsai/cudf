/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "counts.hpp"

#include <cudf/detail/null_mask.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/adjacent_difference.h>

namespace cudf {
namespace reduction {
namespace detail {

rmm::device_uvector<size_type> segmented_counts(bitmask_type const* null_mask,
                                                bool has_nulls,
                                                device_span<size_type const> offsets,
                                                null_policy null_handling,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  auto const num_segments = offsets.size() - 1;

  if (has_nulls && (null_handling == null_policy::EXCLUDE)) {
    return cudf::detail::segmented_count_bits(null_mask,
                                              offsets.begin(),
                                              offsets.end() - 1,
                                              offsets.begin() + 1,
                                              cudf::detail::count_bits_policy::SET_BITS,
                                              stream,
                                              mr);
  }

  rmm::device_uvector<size_type> valid_counts(num_segments, stream, mr);
  thrust::adjacent_difference(
    rmm::exec_policy(stream), offsets.begin() + 1, offsets.end(), valid_counts.begin());
  return valid_counts;
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
