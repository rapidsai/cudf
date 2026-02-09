/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#pragma GCC diagnostic ignored "-Wignored-attributes"  // Work-around for JITIFY2's false-positive
                                                       // warnings when compiled with GCC13

#include <cudf/utilities/export.hpp>

#include <jitify2.hpp>

#include <memory>
#include <mutex>
#include <string>

namespace cudf {
namespace jit {

class program_cache {
  std::mutex _caches_mutex;
  std::unordered_map<std::string, std::unique_ptr<jitify2::ProgramCache<>>> _caches;

 public:
  program_cache()                                = default;
  program_cache(program_cache const&)            = delete;
  program_cache(program_cache&&)                 = delete;
  program_cache& operator=(program_cache const&) = delete;
  program_cache& operator=(program_cache&&)      = delete;
  ~program_cache()                               = default;

  jitify2::ProgramCache<>& get(jitify2::PreprocessedProgramData const& preprog);
};

jitify2::ProgramCache<>& get_program_cache(jitify2::PreprocessedProgramData const& preprog);

}  // namespace jit
}  // namespace cudf
