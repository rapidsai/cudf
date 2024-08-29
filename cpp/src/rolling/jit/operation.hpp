/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "rolling/jit/operation-udf.hpp"

#include <cudf/types.hpp>

struct rolling_udf_ptx {
  template <typename OutType, typename InType>
  static OutType operate(InType const* in_col, cudf::size_type start, cudf::size_type count)
  {
    OutType ret;
    rolling_udf(&ret, 0, 0, 0, 0, &in_col[start], count, sizeof(InType));
    return ret;
  }
};

struct rolling_udf_cuda {
  template <typename OutType, typename InType>
  static OutType operate(InType const* in_col, cudf::size_type start, cudf::size_type count)
  {
    OutType ret;
    rolling_udf(&ret, in_col, start, count);
    return ret;
  }
};
