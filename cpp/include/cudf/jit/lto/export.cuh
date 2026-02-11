/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#define CUDF_LTO_EXPORT __attribute__((visibility("default")))

#define CUDF_LTO_ALIAS __attribute__((may_alias))
