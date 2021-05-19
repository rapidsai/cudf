/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cstdint>
#include <type_traits>

namespace cuco {
namespace detail {
__host__ __device__ constexpr int cuda_memcmp(void const* __lhs, void const* __rhs, size_t __count)
{
  auto __lhs_c = reinterpret_cast<unsigned char const*>(__lhs);
  auto __rhs_c = reinterpret_cast<unsigned char const*>(__rhs);
  while (__count--) {
    auto const __lhs_v = *__lhs_c++;
    auto const __rhs_v = *__rhs_c++;
    if (__lhs_v < __rhs_v) { return -1; }
    if (__lhs_v > __rhs_v) { return 1; }
  }
  return 0;
}

template <std::size_t TypeSize>
struct bitwise_compare_impl {
  __host__ __device__ static constexpr bool compare(char const* lhs, char const* rhs)
  {
    return cuda_memcmp(lhs, rhs, TypeSize) == 0;
  }
};

template <>
struct bitwise_compare_impl<4> {
  __host__ __device__ static constexpr bool compare(char const* lhs, char const* rhs)
  {
    return *reinterpret_cast<uint32_t const*>(lhs) == *reinterpret_cast<uint32_t const*>(rhs);
  }
};

template <>
struct bitwise_compare_impl<8> {
  __host__ __device__ static constexpr bool compare(char const* lhs, char const* rhs)
  {
    return *reinterpret_cast<uint64_t const*>(lhs) == *reinterpret_cast<uint64_t const*>(rhs);
  }
};

/**
 * @brief Performs a bitwise equality comparison between the two specified objects
 *
 * @tparam T Type with unique object representations
 * @param lhs The first object
 * @param rhs The second object
 * @return If the bits in the object representations of lhs and rhs are identical.
 */
template <typename T>
__host__ __device__ constexpr bool bitwise_compare(T const& lhs, T const& rhs)
{
  static_assert(std::has_unique_object_representations_v<T>,
                "Bitwise compared objects must have unique object representation.");
  return detail::bitwise_compare_impl<sizeof(T)>::compare(reinterpret_cast<char const*>(&lhs),
                                                          reinterpret_cast<char const*>(&rhs));
}

}  // namespace detail
}  // namespace cuco