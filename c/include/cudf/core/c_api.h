/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <stdbool.h>
#include <stdint.h>

#if __has_include(<cudf/core/export.h>)
#include <cudf/core/export.h>
#else
#include "export.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief An enum denoting error statuses for function calls.
 */
typedef enum { CUDF_ERROR = 0, CUDF_SUCCESS = 1 } cudfError_t;

/**
 * @brief An opaque C handle for cuDF C API resources.
 */
typedef uintptr_t cudfResources_t;

/**
 * @brief Returns a string describing the last seen error on this thread, or
 *        NULL if the last function succeeded.
 */
CUDF_C_EXPORT const char* cudfGetLastErrorText(void);

/**
 * @brief Sets a string describing an error seen on the thread. Passing NULL
 *        clears any previously seen error message.
 */
CUDF_C_EXPORT void cudfSetLastErrorText(const char* error);

/**
 * @brief Create an initialized opaque resources handle.
 */
CUDF_C_EXPORT cudfError_t cudfResourcesCreate(cudfResources_t* res);

/**
 * @brief Destroy and deallocate an opaque resources handle.
 */
CUDF_C_EXPORT cudfError_t cudfResourcesDestroy(cudfResources_t res);

/**
 * @brief Set cudaStream_t on a resources handle.
 */
CUDF_C_EXPORT cudfError_t cudfStreamSet(cudfResources_t res, cudaStream_t stream);

/**
 * @brief Get the cudaStream_t from a resources handle.
 */
CUDF_C_EXPORT cudfError_t cudfStreamGet(cudfResources_t res, cudaStream_t* stream);

/**
 * @brief Synchronize the current CUDA stream on a resources handle.
 */
CUDF_C_EXPORT cudfError_t cudfStreamSync(cudfResources_t res);

/**
 * @brief Get the version of the cuDF library.
 */
CUDF_C_EXPORT cudfError_t cudfVersionGet(uint16_t* major, uint16_t* minor, uint16_t* patch);

#ifdef __cplusplus
}
#endif
