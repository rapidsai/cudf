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

#include "sha_hash.cuh"

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>

namespace cudf {
namespace hashing {
namespace detail {

namespace {

struct sha224_hash_state {
  uint64_t message_length = 0;
  uint32_t buffer_length  = 0;
  uint32_t hash_value[8]  = {
    0xc1059ed8, 0x367cd507, 0x3070dd17, 0xf70e5939, 0xffc00b31, 0x68581511, 0x64f98fa7, 0xbefa4fa4};
  uint8_t buffer[64];
};

struct SHA224Hash : HashBase<SHA224Hash> {
  __device__ inline SHA224Hash(char* result_location) : HashBase<SHA224Hash>(result_location) {}

  // Intermediate data type storing the hash state
  using hash_state = sha224_hash_state;
  // The word type used by this hash function
  using sha_word_type = uint32_t;
  // Number of bytes processed in each hash step
  static constexpr uint32_t message_chunk_size = 64;
  // Digest size in bytes. This is truncated from SHA-256.
  static constexpr uint32_t digest_size = 56;
  // Number of bytes used for the message length
  static constexpr uint32_t message_length_size = 8;

  __device__ inline void hash_step(hash_state& state) { sha256_hash_step(state); }

  hash_state state;
};

}  // namespace

std::unique_ptr<column> sha224(table_view const& input,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  return sha_hash<SHA224Hash>(input, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> sha224(table_view const& input,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sha224(input, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
