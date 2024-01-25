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
#include <rmm/mr/device/polymorphic_allocator.hpp>

namespace cudf::detail {

class cuco_allocator
  : public rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<char>> {
  using default_allocator = rmm::mr::polymorphic_allocator<char>;
  using base_type         = rmm::mr::stream_allocator_adaptor<default_allocator>;

 public:
  cuco_allocator(rmm::cuda_stream_view stream) : base_type{default_allocator{}, stream} {}
};

}  // namespace cudf::detail
