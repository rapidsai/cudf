/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "nvtx3.hpp"

namespace cudf {
/**
 * @brief Tag type for libcudf's NVTX domain.
 */
struct libcudf_domain {
  static constexpr char const* name{"libcudf"};  ///< Name of the libcudf domain
};

/**
 * @brief Alias for an NVTX range in the libcudf domain.
 */
using thread_range = ::nvtx3::domain_thread_range<libcudf_domain>;

}  // namespace cudf

/**
 * @brief Convenience macro for generating an NVTX range in the `libcudf` domain
 * from the lifetime of a function.
 *
 * Uses the name of the immediately enclosing function returned by `__func__` to
 * name the range.
 *
 * Example:
 * ```
 * void some_function(){
 *    CUDF_FUNC_RANGE();
 *    ...
 * }
 * ```
 */
#define CUDF_FUNC_RANGE_1(F) NVTX3_FUNC_RANGE_IN(cudf::libcudf_domain, F)
#define CUDF_FUNC_RANGE_0()  CUDF_FUNC_RANGE_1(__func__)

#define CUDF_FUNC_RANGE_CHOOSER(_1, NAME, ...) NAME
#define CUDF_FUNC_RANGE_RECOMPOSER(ARGS)       CUDF_FUNC_RANGE_CHOOSER ARGS
#define CUDF_FUNC_RANGE_FROM_ARG_COUNT(...) \
  CUDF_FUNC_RANGE_RECOMPOSER((__VA_ARGS__, CUDF_FUNC_RANGE_1, ))
#define CUDF_FUNC_RANGE_EXPANDER() , CUDF_FUNC_RANGE_0
#define GET_CUDF_FUNC_RANGE_MACRO(...) \
  CUDF_FUNC_RANGE_FROM_ARG_COUNT(CUDF_FUNC_RANGE_EXPANDER __VA_ARGS__())
#define CUDF_FUNC_RANGE(...) GET_CUDF_FUNC_RANGE_MACRO(__VA_ARGS__)(__VA_ARGS__)
