/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Get the number of multiprocessors on the device
 */
cudf::size_type num_multiprocessors();

/**
 * @brief Get the number of elements that can be processed per thread.
 *
 * @param[in] kernel The kernel for which the elements per thread needs to be assessed
 * @param[in] total_size Number of elements
 * @param[in] block_size Expected block size
 *
 * @return cudf::size_type Elements per thread that can be processed for given specification.
 */
template <typename Kernel>
cudf::size_type elements_per_thread(Kernel kernel,
                                    cudf::size_type total_size,
                                    cudf::size_type block_size,
                                    cudf::size_type max_per_thread = 32)
{
  CUDF_FUNC_RANGE();

  // calculate theoretical occupancy
  int max_blocks = 0;
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, block_size, 0));

  int per_thread = total_size / (max_blocks * num_multiprocessors() * block_size);
  return std::clamp(per_thread, 1, max_per_thread);
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
