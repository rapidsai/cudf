/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

namespace cudf {
namespace rolling {
namespace jit {
namespace code {

const char* operation_h =
R"***(operation.h
#pragma once
  struct numba_generic_aggregator {
    template <typename OutType, typename InType>
    static OutType operate(
      const InType* in_col, gdf_index_type start, gdf_index_type count)
    {
      OutType ret;
      NUMBA_GENERIC_AGGREGATOR(
        &ret, 0, 0, 0, 0, &in_col[start], count, sizeof(InType));
      return ret;
    }
  };

  struct cuda_generic_aggregator {
    template <typename OutType, typename InType>
    static OutType operate(
      const InType* in_col, gdf_index_type start, gdf_index_type count)
    {
      OutType ret;
      CUDA_GENERIC_AGGREGATOR(
        &ret, in_col, start, count);
      return ret;
    }
  };

)***";

} // namespace code
} // namespace jit
} // namespace rolling
} // namespace cudf
