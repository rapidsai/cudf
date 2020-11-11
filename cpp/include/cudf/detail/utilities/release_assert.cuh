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

#pragma once

#include <cuda_runtime.h>

/**
 * @brief `assert`-like macro for device code that persists in Release builds.
 *
 * This is effectively the same as the standard `assert` macro, except it will
 * not be compiled out in Release builds, i.e., this macro will be present
 * regardless of the state of `NDEBUG`.
 *
 * Relies on the `__PRETTY_FUNCTION__` macro which is specific to GCC and Clang.
 *
 **/
#if defined(__CUDA_ARCH__) && (defined(__clang__) || defined(__GNUC__))
#define __ASSERT_STR_HELPER(x) #x
#define release_assert(e)     \
  ((e) ? static_cast<void>(0) \
       : __assert_fail(__ASSERT_STR_HELPER(e), __FILE__, __LINE__, __PRETTY_FUNCTION__))
#else
#define release_assert(e) (static_cast<void>(0))
#endif
