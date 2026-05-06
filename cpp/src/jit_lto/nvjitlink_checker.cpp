/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/jit_lto/nvjitlink_checker.hpp>
#include <cudf/utilities/error.hpp>

#include <memory>
#include <nvJitLink.h>
#include <string>

namespace cudf::detail::jit_lto {

void check_nvjitlink_result(nvJitLinkHandle handle, nvJitLinkResult result)
{
  if (result != NVJITLINK_SUCCESS) {
    std::string error_msg = "nvJITLink failed with error " + std::to_string(result);
    size_t log_size       = 0;
    result                = nvJitLinkGetErrorLogSize(handle, &log_size);
    if (result == NVJITLINK_SUCCESS && log_size > 0) {
      std::unique_ptr<char[]> log{new char[log_size]};
      result = nvJitLinkGetErrorLog(handle, log.get());
      if (result == NVJITLINK_SUCCESS) { error_msg += "\n" + std::string(log.get()); }
    }
    CUDF_FAIL(error_msg);
  }
}

}  // namespace cudf::detail::jit_lto
