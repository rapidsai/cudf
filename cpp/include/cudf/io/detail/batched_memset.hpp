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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

/**
 * @brief A helper function that takes in a vector of device spans and memsets them to the
 * value provided using batches sent to the GPU.
 *
 * @param bufs Vector with device spans of data
 * @param value Value to memset all device spans to
 * @param _stream Stream used for device memory operations and kernel launches
 * @param _mr Device memory resource used to allocate the returned column's device memory
 *
 * @return The data in device spans all set to value
 */

namespace CUDF_EXPORT cudf {
namespace io::detail {

void batched_memset(std::vector<cudf::device_span<uint64_t>>& bufs,
                    uint64_t const value,
                    rmm::cuda_stream_view stream);

}  // namespace io::detail
}  // namespace CUDF_EXPORT cudf
