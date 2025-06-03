/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

  jitify2::ProgramCache<>& get(jitify2::PreprocessedProgramData preprog);
};

jitify2::ProgramCache<>& get_program_cache(jitify2::PreprocessedProgramData preprog);

}  // namespace jit
}  // namespace cudf
