/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <strings/regex/regex.cuh>

namespace cudf {
namespace strings {
namespace detail {

/**
 * The stack is used to keep progress (state) on evaluating the regex instructions on each string.
 * So the size of the stack is in proportion to the number of instructions in the given regex
 * pattern.
 *
 * There are four call types based on the number of regex instructions in the given pattern.
 * Small, medium, and large instruction counts can use the stack effectively.
 * Smaller stack sizes execute faster.
 *
 * Patterns with instruction counts bigger than large use global memory rather than the stack
 * for managing the evaluation state data.
 *
 * @tparam Functor The functor to invoke with stack size templated value.
 * @tparam Ts Parameter types for the functor call.
 */
template <typename Functor, typename... Ts>
constexpr decltype(auto) regex_dispatcher(reprog_device d_prog, Functor f, Ts&&... args)
{
  auto const regex_insts = d_prog.insts_counts();
  if (regex_insts <= RX_SMALL_INSTS)
    return f.template operator()<RX_STACK_SMALL>(std::forward<Ts>(args)...);
  if (regex_insts <= RX_MEDIUM_INSTS)
    return f.template operator()<RX_STACK_MEDIUM>(std::forward<Ts>(args)...);
  if (regex_insts <= RX_LARGE_INSTS)
    return f.template operator()<RX_STACK_LARGE>(std::forward<Ts>(args)...);

  return f.template operator()<RX_STACK_ANY>(std::forward<Ts>(args)...);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
