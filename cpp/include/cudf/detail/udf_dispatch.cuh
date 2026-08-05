/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
 
// This header is an inlined header that defines the CUDF_DISPATCH_UDF function. It is placed here
// so the symbols in the headers above can be used by it.
#include <cudf/detail/kernel_instance.cuh>
#include <cudf/detail/operation_udf.cuh>
#include <cudf/detail/udf_expression.cuh>

#ifndef CUDF_LTO_MODE
#define CUDF_UDF_TYPE int()  // Default placeholder for the UDF type.
#endif

// Use LTO-dispatch for transform operators if we're in LTO mode. This allows the operator to be
// defined in a separate translation unit and compiled with LTO, which can result in better
// performance due to more optimization opportunities
#ifdef CUDF_LTO_MODE
#define CUDF_DISPATCH_UDF(...) \
  cudf_udf_symbol(__VA_ARGS__)  // Call the external-linkage LTO-dispatched symbol
#else
#define CUDF_DISPATCH_UDF(...) \
  CUDF_UDF_EXPRESSION(__VA_ARGS__)  // Call the CUDA expression directly
#endif

using cudf_udf_type_t = CUDF_UDF_TYPE;

extern "C" __device__ cudf_udf_type_t cudf_udf_symbol;
