/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/is_element_valid.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

bool is_element_valid_sync(column_view const& col_view,
                           size_type element_index,
                           rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(element_index >= 0 and element_index < col_view.size(), "invalid index.");
  if (!col_view.nullable()) { return true; }

  bitmask_type word = 0;
  // null_mask() returns device ptr to bitmask without offset
  size_type const index = element_index + col_view.offset();
  CUDF_CUDA_TRY(cudaMemcpyAsync(&word,
                                col_view.null_mask() + word_index(index),
                                sizeof(bitmask_type),
                                cudaMemcpyDefault,
                                stream.value()));
  stream.synchronize();
  return static_cast<bool>(word & (bitmask_type{1} << intra_word_index(index)));
}

}  // namespace detail
}  // namespace cudf
