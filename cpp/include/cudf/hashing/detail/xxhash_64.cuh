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

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>

#include <cuco/hash_functions.cuh>
#include <cuda/std/cstddef>

namespace cudf::hashing::detail {

template <typename Key>
struct XXHash_64 : public cuco::xxhash_64<Key> {
  using result_type = typename cuco::xxhash_64<Key>::result_type;

  __host__ __device__ constexpr XXHash_64(uint64_t seed = cudf::DEFAULT_HASH_SEED)
    : cuco::xxhash_64<Key>{seed}
  {
  }

  __device__ result_type operator()(Key const& key) const
  {
    return cuco::xxhash_64<Key>::operator()(key);
  }

  template <typename Extent>
  __device__ result_type compute_hash(cuda::std::byte const* bytes, Extent size) const
  {
    return cuco::xxhash_64<Key>::compute_hash(bytes, size);
  }
};

template <>
XXHash_64<bool>::result_type __device__ inline XXHash_64<bool>::operator()(bool const& key) const
{
  return this->compute_hash(reinterpret_cast<cuda::std::byte const*>(&key), sizeof(key));
}

template <>
XXHash_64<float>::result_type __device__ inline XXHash_64<float>::operator()(float const& key) const
{
  return cuco::xxhash_64<float>::operator()(normalize_nans(key));
}

template <>
XXHash_64<double>::result_type __device__ inline XXHash_64<double>::operator()(
  double const& key) const
{
  return cuco::xxhash_64<double>::operator()(normalize_nans(key));
}

template <>
XXHash_64<cudf::string_view>::result_type
  __device__ inline XXHash_64<cudf::string_view>::operator()(cudf::string_view const& key) const
{
  return this->compute_hash(reinterpret_cast<cuda::std::byte const*>(key.data()), key.size_bytes());
}

template <>
XXHash_64<numeric::decimal32>::result_type
  __device__ inline XXHash_64<numeric::decimal32>::operator()(numeric::decimal32 const& key) const
{
  auto const val = key.value();
  auto const len = sizeof(val);
  return this->compute_hash(reinterpret_cast<cuda::std::byte const*>(&val), len);
}

template <>
XXHash_64<numeric::decimal64>::result_type
  __device__ inline XXHash_64<numeric::decimal64>::operator()(numeric::decimal64 const& key) const
{
  auto const val = key.value();
  auto const len = sizeof(val);
  return this->compute_hash(reinterpret_cast<cuda::std::byte const*>(&val), len);
}

template <>
XXHash_64<numeric::decimal128>::result_type
  __device__ inline XXHash_64<numeric::decimal128>::operator()(numeric::decimal128 const& key) const
{
  auto const val = key.value();
  auto const len = sizeof(val);
  return this->compute_hash(reinterpret_cast<cuda::std::byte const*>(&val), len);
}

}  // namespace cudf::hashing::detail
