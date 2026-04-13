/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "glushkov_regcomp.h"
#include "regcomp.h"
#include "regex.cuh"

#include <cudf/strings/regex/regex_program.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace strings {

/**
 * @brief Implementation object for regex_program
 *
 * It encapsulates internal reprog object used for building its device equivalent.
 * When the GLUSHKOV flag is set and the pattern is eligible, glushkov_prog is
 * non-null and will be used by create_prog_device to construct the device program.
 */
struct regex_program::regex_program_impl {
  detail::reprog prog;
  std::unique_ptr<detail::glushkov_host_program> glushkov_prog;

  regex_program_impl(detail::reprog const& p) : prog(p) {}
  regex_program_impl(detail::reprog&& p) : prog(std::move(p)) {}

  // TODO: There will be other options added here in the future to handle issues
  // 10852 and possibly others like 11979
};

struct regex_device_builder {
  static auto create_prog_device(regex_program const& p,
                                 rmm::cuda_stream_view stream,
                                 bool use_glushkov = true)
  {
    return detail::reprog_device::create(
      p._impl->prog, use_glushkov ? p._impl->glushkov_prog.get() : nullptr, stream);
  }
};

}  // namespace strings
}  // namespace cudf
