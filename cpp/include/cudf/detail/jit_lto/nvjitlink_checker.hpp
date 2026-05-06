/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvJitLink.h>

namespace cudf::detail::jit_lto {

void check_nvjitlink_result(nvJitLinkHandle handle, nvJitLinkResult result);

}  // namespace cudf::detail::jit_lto
