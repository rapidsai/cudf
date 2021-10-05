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
#include <cudf/detail/nvtx/nvtx3.hpp>
#include <cudf/utilities/error.hpp>
#include <thrust/device_ptr.h>
#include <thrust/equal.h>

#include "check_output_sizes.hpp"

namespace {

struct java_domain {
  static constexpr char const *name{"Java"};
};

} // anonymous namespace

namespace cudf {
namespace java {

/**
 * Copy a simple vector to device memory asynchronously. Be sure to read
 * the data on the same stream as is used to copy it.
 */
bool check_nvcomp_output_sizes(std::size_t const *uncompressed_sizes,
                               std::size_t const *actual_uncompressed_sizes, std::size_t batch_size,
                               rmm::cuda_stream_view stream) {
  NVTX3_FUNC_RANGE_IN(java_domain);
  thrust::device_ptr<const size_t> dev_uncompressed_sizes(uncompressed_sizes);
  thrust::device_ptr<const size_t> dev_actual_uncompressed_sizes(actual_uncompressed_sizes);
  return thrust::equal(rmm::exec_policy(stream), dev_uncompressed_sizes,
                       dev_uncompressed_sizes + batch_size, dev_actual_uncompressed_sizes);
}

} // namespace java
} // namespace cudf
