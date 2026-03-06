/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#pragma GCC diagnostic ignored "-Wignored-attributes"  // Work-around for JITIFY2's false-positive
                                                       // warnings when compiled with GCC13

#include <cudf/utilities/export.hpp>

#include <jitify2.hpp>

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

namespace cudf {
namespace jit {

class program_cache {
  std::mutex _caches_mutex;
  std::unordered_map<std::string, std::unique_ptr<jitify2::ProgramCache<>>> _caches;
  int32_t _kernel_limit_proc;
  int32_t _kernel_limit_disk;
  std::filesystem::path _cache_dir;
  std::atomic<bool> _disabled;

 public:
  program_cache(int32_t kernel_limit_proc,
                int32_t kernel_limit_disk,
                std::filesystem::path cache_dir,
                bool disabled)
    : _kernel_limit_proc(kernel_limit_proc),
      _kernel_limit_disk(kernel_limit_disk),
      _cache_dir(std::move(cache_dir)),
      _disabled(disabled)
  {
  }

  program_cache(program_cache const&)            = delete;
  program_cache(program_cache&&)                 = delete;
  program_cache& operator=(program_cache const&) = delete;
  program_cache& operator=(program_cache&&)      = delete;
  ~program_cache()                               = default;

  jitify2::ProgramCache<>& get(jitify2::PreprocessedProgramData const& preprog);

  void clear();

  void enable(bool enable);

  bool is_enabled() const;

  static std::unique_ptr<jit::program_cache> create();
};

jitify2::ProgramCache<>& get_program_cache(jitify2::PreprocessedProgramData const& preprog);

}  // namespace jit
}  // namespace cudf
