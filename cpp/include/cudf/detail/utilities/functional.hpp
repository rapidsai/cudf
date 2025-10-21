/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/functional>
#include <thrust/functional.h>

namespace cudf::detail {

#if CCCL_MAJOR_VERSION >= 3
using cuda::maximum;
using cuda::minimum;
#else
using thrust::maximum;
using thrust::minimum;
#endif

}  // namespace cudf::detail
