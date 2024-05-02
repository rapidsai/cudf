/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

/**
 * @brief A casting functor wrapping another functor.
 * @file
 */

#include <cudf/types.hpp>

#include <cuda/functional>

#include <type_traits>
#include <utility>

namespace cudf {
namespace detail {

/**
 * @brief Functor that casts another functor's result to a specified type.
 *
 * CUB 2.0.0 reductions require that the binary operator returns the same type
 * as the initial value type, so we wrap binary operators with this when used
 * by CUB.
 */
template <typename ResultType, typename F>
struct cast_functor_fn {
  F f;

  template <typename... Ts>
  CUDF_HOST_DEVICE inline ResultType operator()(Ts&&... args)
  {
    return static_cast<ResultType>(f(std::forward<Ts>(args)...));
  }
};

/**
 * @brief Function creating a casting functor.
 */
template <typename ResultType, typename F>
inline cast_functor_fn<ResultType, std::decay_t<F>> cast_functor(F&& f)
{
  return cast_functor_fn<ResultType, std::decay_t<F>>{std::forward<F>(f)};
}

}  // namespace detail

}  // namespace cudf
