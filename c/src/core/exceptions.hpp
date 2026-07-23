/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/core/c_api.h>
#include <cudf/utilities/error.hpp>

#include <exception>
#include <string>

namespace cudf::c {

/**
 * @brief Translates C++ exceptions into cuDF C API error codes.
 */
template <typename Fn>
cudfError_t translate_exceptions(Fn func)
{
  cudfError_t status;
  try {
    func();
    status = CUDF_SUCCESS;
    cudfSetLastErrorText(nullptr);
  } catch (cudf::fatal_cuda_error const& e) {
    auto const what =
      std::string{e.what() == nullptr ? "" : e.what()} + " [FATAL CUDA ERROR - handle poisoned]";
    cudfSetLastErrorText(what.c_str());
    status = CUDF_ERROR;
  } catch (cudf::cuda_error const& e) {
    cudfSetLastErrorText(e.what());
    status = CUDF_ERROR;
  } catch (cudf::logic_error const& e) {
    cudfSetLastErrorText(e.what());
    status = CUDF_ERROR;
  } catch (std::exception const& e) {
    cudfSetLastErrorText(e.what());
    status = CUDF_ERROR;
  } catch (...) {
    cudfSetLastErrorText("unknown exception");
    status = CUDF_ERROR;
  }
  return status;
}

}  // namespace cudf::c
