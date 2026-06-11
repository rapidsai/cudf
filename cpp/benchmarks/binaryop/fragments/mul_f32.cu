/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ int transform(int32_t* out, int32_t a, int32_t b) { *out = a * b; return 0; }
