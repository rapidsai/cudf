/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "regcomp.h"
#include "regex.cuh"

#include <cudf/strings/regex/regex_program.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {

/**
 * @brief Implementation object for regex_program
 *
 * It encapsulates internal reprog object used for building its device equivalent
 */
struct regex_program::regex_program_impl {
  detail::reprog prog;

  regex_program_impl(detail::reprog const& p) : prog(p) {}
  regex_program_impl(detail::reprog&& p) : prog(p) {}

  // TODO: There will be other options added here in the future to handle issues
  // 10852 and possibly others like 11979
};

struct regex_device_builder {
  static auto create_prog_device(regex_program const& p, rmm::cuda_stream_view stream)
  {
    return detail::reprog_device::create(p._impl->prog, stream);
  }
};

}  // namespace strings
}  // namespace cudf
