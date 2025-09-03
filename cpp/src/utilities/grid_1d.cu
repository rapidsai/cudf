/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf::detail {

grid_1d::grid_1d(thread_index_type overall_num_elements,
                 thread_index_type num_threads_per_block,
                 thread_index_type elements_per_thread)
  : num_threads_per_block(num_threads_per_block),
    num_blocks(
      util::div_rounding_up_safe(overall_num_elements, elements_per_thread * num_threads_per_block))
{
  CUDF_EXPECTS(num_threads_per_block > 0, "num_threads_per_block must be > 0");
  CUDF_EXPECTS(num_blocks > 0, "num_blocks must be > 0");
}

}  // namespace cudf::detail
