/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace hashing {
namespace detail {

std::unique_ptr<column> hash(table_view const& input,
                             hash_id hash_function,
                             uint32_t seed,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  switch (hash_function) {
    case (hash_id::HASH_MURMUR3): return murmurhash3_x86_32(input, seed, stream, mr);
    case (hash_id::HASH_SPARK_MURMUR3): return spark_murmurhash3_x86_32(input, seed, stream, mr);
    case (hash_id::HASH_MD5): return md5(input, stream, mr);
    default: CUDF_FAIL("Unsupported hash function.");
  }
}

}  // namespace detail
}  // namespace hashing

std::unique_ptr<column> hash(table_view const& input,
                             hash_id hash_function,
                             uint32_t seed,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return hashing::detail::hash(input, hash_function, seed, stream, mr);
}

}  // namespace cudf
