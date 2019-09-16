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

#include <bitmask/legacy/bit_mask.cuh>

#include <cub/cub.cuh>

using bit_mask::bit_mask_t;

namespace cudf
{

namespace detail
{

struct column_range_factory {
  gdf_column column;
  gdf_index_type begin;

  template <typename T>
  struct column_range {
    T const * column_data;
    bit_mask_t const * bitmask;
    gdf_index_type begin;

    __device__
    T data(gdf_index_type index) { 
      return column_data[begin + index]; }

    __device__
    bool valid(gdf_index_type index) {
      return bit_mask::is_valid(bitmask, begin + index);
    }
  };

  template <typename T>
  column_range<T> make() {
    return column_range<T>{
      static_cast<T*>(column.data),
      reinterpret_cast<bit_mask_t*>(column.valid),
      begin
    };
  }
};

} // namespace cudf

} // namespace detail
