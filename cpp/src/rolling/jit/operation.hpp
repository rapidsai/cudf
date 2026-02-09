/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <rolling/jit/operation-udf.hpp>

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
