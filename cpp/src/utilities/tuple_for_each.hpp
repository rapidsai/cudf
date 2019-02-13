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
#ifndef TUPLE_FOR_EACH_H
#define TUPLE_FOR_EACH_H

namespace cudf {
namespace detail {
template <typename Tuple, typename F, std::size_t... Indices>
inline void for_each_impl(Tuple&& tuple, F&& f,
                          std::index_sequence<Indices...>) {
  using swallow = int[];
  (void)swallow{
      1, (f(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})...};
}

/**---------------------------------------------------------------------------*
 * @brief A `for_each` over the elements of a tuple.
 *
 * For every element in a tuple, invokes a unary callable and passes the tuple
 * element into the callable.
 *
 * @tparam Tuple The type of the tuple
 * @tparam F The type of the callable
 * @param tuple The tuple to iterate over
 * @param f The unary callable
 *---------------------------------------------------------------------------**/
template <typename Tuple, typename F>
inline void for_each(Tuple&& tuple, F&& f) {
  constexpr std::size_t N =
      std::tuple_size<std::remove_reference_t<Tuple>>::value;
  for_each_impl(std::forward<Tuple>(tuple), std::forward<F>(f),
                std::make_index_sequence<N>{});
}
}  // namespace detail
}  // namespace cudf
#endif