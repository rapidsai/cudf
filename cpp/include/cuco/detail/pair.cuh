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

#include <thrust/pair.h>

namespace cuco {
namespace detail {

/**
 * @brief Rounds `v` to the nearest power of 2 greater than or equal to `v`. 
 * 
 * @param v 
 * @return The nearest power of 2 greater than or equal to `v`.
 */
constexpr std::size_t next_pow2(std::size_t v) noexcept {
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return ++v;
}

/**
 * @brief Gives value to use as alignment for a pair type that is at least the
 * size of the sum of the size of the first type and second type, or 16,
 * whichever is smaller.
 */
template <typename First, typename Second>
constexpr std::size_t pair_alignment() {
  return std::min(std::size_t{16}, next_pow2(sizeof(First) + sizeof(Second)));
}
} // namespace detail

/**
 * @brief Custom pair type
 *
 * This is necessary because `thrust::pair` is under aligned.
 *
 * @tparam First
 * @tparam Second
 */
template <typename First, typename Second>
struct alignas(detail::pair_alignment<First, Second>()) pair {
  using first_type = First;
  using second_type = Second;
  First first{};
  Second second{};
  pair() = default;
  __host__ __device__ constexpr pair(thrust::pair<First, Second> const& p) noexcept
    : first{p.first}, second{p.second}
  {
  }
};

template <typename K, typename V>
using pair_type = cuco::pair<K, V>;

/**
 * @brief Creates a pair of type `pair_type`
 * 
 * @tparam F 
 * @tparam S 
 * @param f 
 * @param s 
 * @return pair_type with first element `f` and second element `s`.
 */
template <typename F, typename S>
__host__ __device__ pair_type<F, S> make_pair(F&& f, S&& s) noexcept {
  return pair_type<F, S>{std::forward<F>(f), std::forward<S>(s)};
}
} // namespace cuco