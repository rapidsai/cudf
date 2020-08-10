/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

namespace cudf {
namespace detail {
std::unique_ptr<column> mask_to_bools(bitmask_type const* bitmask,
                                      size_type begin_bit,
                                      size_type end_bit,
                                      cudaStream_t stream,
                                      rmm::mr::device_memory_resource* mr)
{
  auto const length = end_bit - begin_bit;
  CUDF_EXPECTS(length >= 0, "begin_bit should be less than or equal to end_bit");
  CUDF_EXPECTS((bitmask != nullptr) or (length == 0), "nullmask is null");

  auto out_col =
    make_fixed_width_column(data_type(type_id::BOOL8), length, mask_state::UNALLOCATED, stream, mr);

  if (length > 0) {
    auto mutable_view = out_col->mutable_view();

    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      thrust::make_counting_iterator<cudf::size_type>(begin_bit),
                      thrust::make_counting_iterator<cudf::size_type>(end_bit),
                      mutable_view.begin<bool>(),
                      [bitmask] __device__(auto index) { return bit_is_set(bitmask, index); });
  }

  return out_col;
}
}  // namespace detail

std::unique_ptr<column> mask_to_bools(bitmask_type const* bitmask,
                                      size_type begin_bit,
                                      size_type end_bit,
                                      rmm::mr::device_memory_resource* mr)
{
  return detail::mask_to_bools(bitmask, begin_bit, end_bit, 0, mr);
}
}  // namespace cudf
