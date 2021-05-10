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

#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/column_factories.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

namespace cudf {
namespace lists {
namespace detail {

std::unique_ptr<cudf::column> make_lists_column_from_scalar(list_scalar const& value,
                                                            size_type size,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::mr::device_memory_resource* mr)
{
  // Handcraft a 1-row column
  auto offsets   = make_numeric_column(data_type(type_id::INT32), 2, mask_state::UNALLOCATED);
  auto m_offsets = offsets->mutable_view();
  thrust::sequence(rmm::exec_policy(stream),
                   m_offsets.begin<size_type>(),
                   m_offsets.end<size_type>(),
                   0,
                   value.view().size());
  auto child           = std::make_unique<column>(value.view());
  size_type null_count = value.is_valid(stream) ? 0 : 1;
  auto null_mask       = null_count ? create_null_mask(1, mask_state::ALL_NULL)
                              : create_null_mask(1, mask_state::UNALLOCATED);
  if (size == 1) {
    return make_lists_column(
      1, std::move(offsets), std::move(child), null_count, std::move(null_mask), stream, mr);
  }

  auto one_row_col = make_lists_column(
    1, std::move(offsets), std::move(child), null_count, std::move(null_mask), stream);

  auto begin = thrust::make_constant_iterator(0);
  auto res   = cudf::detail::gather(table_view({one_row_col->view()}),
                                  begin,
                                  begin + size,
                                  out_of_bounds_policy::DONT_CHECK,
                                  stream,
                                  mr);
  return std::move(res->release()[0]);
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
