

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/utilities/export.hpp>

#include <jit/rtc/rtc.hpp>

namespace CUDF_EXPORT cudf {
namespace rtc {

kernel_ref compile_and_link_udf(char const* name,
                                char const* kernel_name,
                                char const* kernel_key,
                                char const* udf_code,
                                char const* udf_key);

}  // namespace rtc
}  // namespace CUDF_EXPORT cudf
