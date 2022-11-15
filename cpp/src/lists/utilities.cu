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

#include "utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/labeling/label_segments.cuh>

namespace cudf::lists::detail {

std::unique_ptr<column> generate_labels(lists_column_view const& input,
                                        size_type n_elements,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  auto labels = make_numeric_column(
    data_type(type_to_id<size_type>()), n_elements, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const labels_begin = labels->mutable_view().template begin<size_type>();
  cudf::detail::label_segments(
    input.offsets_begin(), input.offsets_end(), labels_begin, labels_begin + n_elements, stream);
  return labels;
}

std::unique_ptr<column> reconstruct_offsets(column_view const& labels,
                                            size_type n_lists,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr)

{
  auto out_offsets = make_numeric_column(
    data_type{type_to_id<offset_type>()}, n_lists + 1, mask_state::UNALLOCATED, stream, mr);

  auto const labels_begin  = labels.template begin<size_type>();
  auto const offsets_begin = out_offsets->mutable_view().template begin<offset_type>();
  cudf::detail::labels_to_offsets(labels_begin,
                                  labels_begin + labels.size(),
                                  offsets_begin,
                                  offsets_begin + out_offsets->size(),
                                  stream);
  return out_offsets;
}

}  // namespace cudf::lists::detail
