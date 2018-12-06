/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

/** ---------------------------------------------------------------------------*
 * @brief Device Memory Manager public C++ interface. 
 * 
 * Efficient allocation, deallocation and tracking of GPU memory.
 * --------------------------------------------------------------------------**/

namespace rmm
{
  template <typename T>
  rmmError_t alloc(T **ptr, size_t size, cudaStream_t stream, const char* file, unsigned int line);

  template <typename T>
  rmmError_t realloc(T **ptr, size_t new_size, cudaStream_t stream, const char* file, unsigned int line);
}
