/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Macros used for defining symbol visibility, only GLIBC is supported
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define CUDF_EXPORT __attribute__((visibility("default")))
#define CUDF_HIDDEN __attribute__((visibility("hidden")))
#else
#define CUDF_EXPORT
#define CUDF_HIDDEN
#endif
