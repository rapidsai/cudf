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

#include <cudf/hashing.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <functional>

namespace CUDF_EXPORT cudf {
namespace hashing::detail {

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref mr);

std::unique_ptr<table> murmurhash3_x64_128(table_view const& input,
                                           uint64_t seed,
                                           rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref mr);

std::unique_ptr<column> md5(table_view const& input,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

std::unique_ptr<column> sha1(table_view const& input,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr);

std::unique_ptr<column> sha224(table_view const& input,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

std::unique_ptr<column> sha256(table_view const& input,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

std::unique_ptr<column> sha384(table_view const& input,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

std::unique_ptr<column> sha512(table_view const& input,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr);

std::unique_ptr<column> xxhash_64(table_view const& input,
                                  uint64_t seed,
                                  rmm::cuda_stream_view,
                                  rmm::device_async_resource_ref mr);

/* Copyright 2005-2014 Daniel James.
 *
 * Use, modification and distribution is subject to the Boost Software
 * License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */
/**
 * @brief Combines two hash values into a single hash value.
 *
 * Taken from the Boost hash_combine function.
 * https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
 *
 * @param lhs The first hash value
 * @param rhs The second hash value
 * @return Combined hash value
 */
CUDF_HOST_DEVICE constexpr uint32_t hash_combine(uint32_t lhs, uint32_t rhs)
{
  return lhs ^ (rhs + 0x9e37'79b9 + (lhs << 6) + (lhs >> 2));
}

/* Copyright 2005-2014 Daniel James.
 *
 * Use, modification and distribution is subject to the Boost Software
 * License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */
/**
 * @brief Combines two hash values into a single hash value.
 *
 * Adapted from Boost hash_combine function and modified for 64-bit.
 * https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
 *
 * @param lhs The first hash value
 * @param rhs The second hash value
 * @return Combined hash value
 */
constexpr std::size_t hash_combine(std::size_t lhs, std::size_t rhs)
{
  return lhs ^ (rhs + 0x9e37'79b9'7f4a'7c15 + (lhs << 6) + (lhs >> 2));
}

}  // namespace hashing::detail
}  // namespace CUDF_EXPORT cudf

// specialization of std::hash for cudf::data_type
namespace std {
template <>
struct hash<cudf::data_type> {
  std::size_t operator()(cudf::data_type const& type) const noexcept
  {
    return cudf::hashing::detail::hash_combine(
      std::hash<int32_t>{}(static_cast<int32_t>(type.id())), std::hash<int32_t>{}(type.scale()));
  }
};
}  // namespace std
