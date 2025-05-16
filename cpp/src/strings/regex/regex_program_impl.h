/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
