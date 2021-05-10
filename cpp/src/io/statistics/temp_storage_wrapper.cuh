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

#include <cudf/strings/string_view.cuh>

#include <cudf/fixed_point/fixed_point.hpp>

#include <cudf/wrappers/durations.hpp>

#include <cudf/wrappers/timestamps.hpp>

#include "statistics.cuh"

#include <cub/cub.cuh>

namespace cudf {
namespace io {
namespace detail {

template <typename T, int block_size>
using cub_temp_storage = typename cub::BlockReduce<T, block_size>::TempStorage;

#define MEMBER_NAME(TYPE) TYPE##_stats

#define DECLARE_MEMBER(TYPE) cub_temp_storage<TYPE, block_size> MEMBER_NAME(TYPE);

template <int block_size>
union block_reduce_storage {
  DECLARE_MEMBER(bool)
  DECLARE_MEMBER(int8_t)
  DECLARE_MEMBER(int16_t)
  DECLARE_MEMBER(int32_t)
  DECLARE_MEMBER(int64_t)
  DECLARE_MEMBER(uint8_t)
  DECLARE_MEMBER(uint16_t)
  DECLARE_MEMBER(uint32_t)
  DECLARE_MEMBER(uint64_t)
  DECLARE_MEMBER(float)
  DECLARE_MEMBER(double)
  DECLARE_MEMBER(string_view)
};

template <typename T, int block_size>
struct temp_storage_wrapper {
  block_reduce_storage<block_size>& storage;
  __device__ temp_storage_wrapper(block_reduce_storage<block_size>& _temp_storage)
    : storage(_temp_storage)
  {
  }
  __device__ cub_temp_storage<T, block_size>& get() = delete;
};

#define TEMP_STORAGE_WRAPPER(TYPE)                                                             \
  template <int block_size>                                                                    \
  struct temp_storage_wrapper<TYPE, block_size> {                                              \
    block_reduce_storage<block_size>& storage;                                                 \
    __device__ temp_storage_wrapper(block_reduce_storage<block_size>& _temp_storage)           \
      : storage(_temp_storage)                                                                 \
    {                                                                                          \
    }                                                                                          \
    __device__ cub_temp_storage<TYPE, block_size>& get() { return storage.MEMBER_NAME(TYPE); } \
  };

TEMP_STORAGE_WRAPPER(bool);
TEMP_STORAGE_WRAPPER(int8_t);
TEMP_STORAGE_WRAPPER(int16_t);
TEMP_STORAGE_WRAPPER(int32_t);
TEMP_STORAGE_WRAPPER(int64_t);
TEMP_STORAGE_WRAPPER(uint8_t);
TEMP_STORAGE_WRAPPER(uint16_t);
TEMP_STORAGE_WRAPPER(uint32_t);
TEMP_STORAGE_WRAPPER(uint64_t);
TEMP_STORAGE_WRAPPER(float);
TEMP_STORAGE_WRAPPER(double);
TEMP_STORAGE_WRAPPER(string_view);

#define STORAGE_WRAPPER_GET(TYPE)                                                                 \
  template <typename T>                                                                           \
  __device__ std::enable_if_t<std::is_same_v<T, TYPE>, cub_temp_storage<TYPE, block_size>&> get() \
  {                                                                                               \
    return storage.MEMBER_NAME(TYPE);                                                             \
  }

template <int block_size>
struct storage_wrapper {
  block_reduce_storage<block_size>& storage;
  __device__ storage_wrapper(block_reduce_storage<block_size>& _temp_storage)
    : storage(_temp_storage)
  {
  }

  STORAGE_WRAPPER_GET(bool);
  STORAGE_WRAPPER_GET(int8_t);
  STORAGE_WRAPPER_GET(int16_t);
  STORAGE_WRAPPER_GET(int32_t);
  STORAGE_WRAPPER_GET(int64_t);
  STORAGE_WRAPPER_GET(uint8_t);
  STORAGE_WRAPPER_GET(uint16_t);
  STORAGE_WRAPPER_GET(uint32_t);
  STORAGE_WRAPPER_GET(uint64_t);
  STORAGE_WRAPPER_GET(float);
  STORAGE_WRAPPER_GET(double);
  STORAGE_WRAPPER_GET(string_view);
};

#undef TEMP_STORAGE_WRAPPER
#undef DECLARE_MEMBER
#undef MEMBER_NAME

}  // namespace detail
}  // namespace io
}  // namespace cudf
