/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf/strings/udf/udf_string.hpp>

namespace cudf::strings::udf {

/**
 * @brief Container for a udf_string and its NRT memory information
 *
 * `meminfo` is a MemInfo struct from numba-cuda, see:
 * https://github.com/NVIDIA/numba-cuda/blob/main/numba_cuda/numba/cuda/memory_management/nrt.cuh
 */
struct managed_udf_string {
  void* meminfo;
  cudf::strings::udf::udf_string udf_str;
};

}  // namespace cudf::strings::udf
