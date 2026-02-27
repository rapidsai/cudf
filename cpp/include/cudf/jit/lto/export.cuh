/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if (defined(__GNUC__) && !defined(__MINGW32__) && !defined(__MINGW64__))

#define CUDF_LTO_EXPORT __attribute__((visibility("default")))

#else

#define CUDF_LTO_EXPORT

#endif

#define CUDF_LTO_ALIAS __attribute__((may_alias))
