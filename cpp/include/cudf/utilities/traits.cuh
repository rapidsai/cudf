/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/atomic>

namespace cudf {

/**
 * @addtogroup utility_types
 * @{
 * @file
 */

/**
 * @brief Indicates whether the type `T` has support for atomics
 *
 * @tparam T     The type to verify
 * @return true  `T` has support for atomics
 * @return false `T` no support for atomics
 */
template <typename T>
constexpr inline bool has_atomic_support()
{
  return cuda::std::atomic<T>::is_always_lock_free;
}

struct has_atomic_support_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return has_atomic_support<T>();
  }
};

/**
 * @brief Indicates whether `type` has support for atomics
 *
 * @param type   The `data_type` to verify
 * @return true  `type` has support for atomics
 * @return false `type` no support for atomics
 */
constexpr inline bool has_atomic_support(data_type type)
{
  return cudf::type_dispatcher(type, has_atomic_support_impl{});
}

/** @} */

}  // namespace cudf
