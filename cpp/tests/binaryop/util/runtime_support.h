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

#include <cuda_runtime.h>

inline bool can_do_runtime_jit()
{
  // We require a CUDA NVRTC of 11.5+ to do runtime jit
  // as we need support for __int128

  int runtime      = 0;
  auto error_value = cudaRuntimeGetVersion(&runtime);
  return (error_value == cudaSuccess) && (runtime >= 11050);
}
