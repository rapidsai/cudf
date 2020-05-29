/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/filling.hpp>
#include <cudf/types.hpp>

#include <memory>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::fill_in_place
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void fill_in_place(mutable_column_view& destination,
                   size_type begin,
                   size_type end,
                   scalar const& value,
                   cudaStream_t stream = 0);

/**
 * @copydoc cudf::fill
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> fill(column_view const& input,
                             size_type begin,
                             size_type end,
                             scalar const& value,
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                             cudaStream_t stream                 = 0);

}  // namespace detail
}  // namespace cudf
