/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>

#include <nvbench/nvbench.cuh>

enum class rank_method : int32_t {};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::rank_method,
  [](cudf::rank_method value) {
    switch (value) {
      case cudf::rank_method::FIRST: return "FIRST";
      case cudf::rank_method::AVERAGE: return "AVERAGE";
      case cudf::rank_method::MIN: return "MIN";
      case cudf::rank_method::MAX: return "MAX";
      case cudf::rank_method::DENSE: return "DENSE";
      default: return "unknown";
    }
  },
  [](cudf::rank_method value) {
    switch (value) {
      case cudf::rank_method::FIRST: return "cudf::rank_method::FIRST";
      case cudf::rank_method::AVERAGE: return "cudf::rank_method::AVERAGE";
      case cudf::rank_method::MIN: return "cudf::rank_method::MIN";
      case cudf::rank_method::MAX: return "cudf::rank_method::MAX";
      case cudf::rank_method::DENSE: return "cudf::rank_method::DENSE";
      default: return "unknown";
    }
  })

using methods = nvbench::enum_type_list<cudf::rank_method::AVERAGE,
                                        cudf::rank_method::DENSE,
                                        cudf::rank_method::FIRST,
                                        cudf::rank_method::MAX,
                                        cudf::rank_method::MIN>;
