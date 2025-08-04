/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "config.hpp"

#include <cstdlib>
#include <string>

namespace cudf {
namespace jit {

jit_config& jit_config::instance()
{
  static jit_config config;
  static bool initialized = false;
  if (!initialized) {
    config.load_from_environment();
    initialized = true;
  }
  return config;
}

jit_config::jit_config() = default;

void jit_config::set_compilation_mode(compilation_mode mode)
{
  _mode = mode;
}

jit_config::compilation_mode jit_config::get_compilation_mode() const
{
  return _mode;
}

bool jit_config::is_lto_ir_enabled() const
{
#ifdef CUDF_USE_LTO_IR
  return _mode == compilation_mode::AUTO ||
         _mode == compilation_mode::LTO_IR_ONLY ||
         _mode == compilation_mode::PREFER_LTO_IR;
#else
  return false;
#endif
}

bool jit_config::is_cuda_fallback_allowed() const
{
  return _mode == compilation_mode::AUTO ||
         _mode == compilation_mode::CUDA_ONLY ||
         _mode == compilation_mode::PREFER_LTO_IR;
}

void jit_config::set_aggressive_operator_detection(bool aggressive)
{
  _aggressive_detection = aggressive;
}

bool jit_config::is_aggressive_operator_detection_enabled() const
{
  return _aggressive_detection;
}

void jit_config::load_from_environment()
{
  // Load compilation mode from environment
  auto mode_env = std::getenv("CUDF_JIT_COMPILATION_MODE");
  if (mode_env != nullptr) {
    std::string mode_str(mode_env);
    if (mode_str == "auto") {
      _mode = compilation_mode::AUTO;
    } else if (mode_str == "lto_ir_only") {
      _mode = compilation_mode::LTO_IR_ONLY;
    } else if (mode_str == "cuda_only") {
      _mode = compilation_mode::CUDA_ONLY;
    } else if (mode_str == "prefer_lto_ir") {
      _mode = compilation_mode::PREFER_LTO_IR;
    }
    // If invalid mode, keep default (AUTO)
  }

  // Load aggressive detection setting
  auto aggressive_env = std::getenv("CUDF_JIT_AGGRESSIVE_DETECTION");
  if (aggressive_env != nullptr) {
    std::string aggressive_str(aggressive_env);
    _aggressive_detection = (aggressive_str == "true" || aggressive_str == "1");
  }
}

}  // namespace jit
}  // namespace cudf