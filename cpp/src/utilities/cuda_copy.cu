/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/cuda_copy.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>

namespace cudf::detail::impl {

void copy_pinned(void* dst, void const* src, std::size_t size, rmm::cuda_stream_view stream)
{
  if (size == 0) return;

  if (size < get_kernel_pinned_copy_threshold()) {
    thrust::copy_n(rmm::exec_policy_nosync(stream),
                   static_cast<const char*>(src),
                   size,
                   static_cast<char*>(dst));
  } else {
    CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
  }
}

}  // namespace cudf::detail::impl
