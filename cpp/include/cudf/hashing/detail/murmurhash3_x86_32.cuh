/*
 * Copyright (c) 2017-2024, NVIDIA CORPORATION.
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

#include "hash_functions.cuh"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/lists/list_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/struct_view.hpp>
#include <cudf/types.hpp>

#include <cuco/hash_functions.cuh>
#include <cuda/std/cstddef>

namespace cudf::hashing::detail {

template <typename Key>
struct MurmurHash3_x86_32 : public cuco::murmurhash3_32<Key> {
  using result_type = typename cuco::murmurhash3_32<Key>::result_type;

  __host__ __device__ constexpr MurmurHash3_x86_32(uint32_t seed = cudf::DEFAULT_HASH_SEED)
    : cuco::murmurhash3_32<Key>{seed}
  {
  }

  __device__ result_type operator()(Key const& key) const
  {
    return cuco::murmurhash3_32<Key>::operator()(key);
  }

  template <typename Extent>
  __device__ result_type compute_hash(cuda::std::byte const* bytes, Extent size) const
  {
    return cuco::murmurhash3_32<Key>::compute_hash(bytes, size);
  }
};

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<bool>::operator()(bool const& key) const
{
  return this->compute_hash(reinterpret_cast<cuda::std::byte const*>(&key), sizeof(key));
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<float>::operator()(float const& key) const
{
  return cuco::murmurhash3_32<float>::operator()(normalize_nans_and_zeros(key));
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<double>::operator()(double const& key) const
{
  return cuco::murmurhash3_32<double>::operator()(normalize_nans_and_zeros(key));
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  return this->compute_hash(reinterpret_cast<cuda::std::byte const*>(key.data()), key.size_bytes());
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{
  auto const val = key.value();
  auto const len = sizeof(val);
  return this->compute_hash(reinterpret_cast<cuda::std::byte const*>(&val), len);
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{
  auto const val = key.value();
  auto const len = sizeof(val);
  return this->compute_hash(reinterpret_cast<cuda::std::byte const*>(&val), len);
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const
{
  auto const val = key.value();
  auto const len = sizeof(val);
  return this->compute_hash(reinterpret_cast<cuda::std::byte const*>(&val), len);
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<cudf::list_view>::operator()(
  cudf::list_view const& key) const
{
  CUDF_UNREACHABLE("List column hashing is not supported");
}

template <>
hash_value_type __device__ inline MurmurHash3_x86_32<cudf::struct_view>::operator()(
  cudf::struct_view const& key) const
{
  CUDF_UNREACHABLE("Direct hashing of struct_view is not supported");
}

}  // namespace cudf::hashing::detail
