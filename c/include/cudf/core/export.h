/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))
#define CUDF_C_EXPORT __attribute__((visibility("default")))
#define CUDF_C_HIDDEN __attribute__((visibility("hidden")))
#else
#define CUDF_C_EXPORT
#define CUDF_C_HIDDEN
#endif
