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

#pragma once

#include <rmm/cuda_stream_view.hpp>

namespace cudf::detail {

namespace impl {

void copy_pinned(void* dst, void const* src, size_t size, rmm::cuda_stream_view stream);

}  // namespace impl

template <typename T>
void copy_pinned_to_device_async(T* dst, T const* src, size_t size, rmm::cuda_stream_view stream)
{
  impl::copy_pinned(dst, src, size * sizeof(T), stream);
}

template <typename T>
void copy_device_to_pinned_async(T* dst, T const* src, size_t size, rmm::cuda_stream_view stream)
{
  impl::copy_pinned(dst, src, size * sizeof(T), stream);
}

}  // namespace cudf::detail
