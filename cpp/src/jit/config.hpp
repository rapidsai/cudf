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

#pragma once

#include <cudf/utilities/export.hpp>

#include <string>

namespace cudf {
namespace jit {

/**
 * @brief Configuration settings for JIT compilation
 */
class jit_config {
 public:
  /**
   * @brief JIT compilation modes
   */
  enum class compilation_mode {
    AUTO,        ///< Automatically choose based on availability and performance
    LTO_IR_ONLY, ///< Use only LTO-IR compilation, fail if not available  
    CUDA_ONLY,   ///< Use only traditional CUDA C++ compilation
    PREFER_LTO_IR ///< Prefer LTO-IR but fall back to CUDA C++ if needed
  };

  /**
   * @brief Get the global JIT configuration instance
   */
  static jit_config& instance();

  /**
   * @brief Set the compilation mode
   * 
   * @param mode The compilation mode to use
   */
  void set_compilation_mode(compilation_mode mode);

  /**
   * @brief Get the current compilation mode
   * 
   * @return The current compilation mode
   */
  compilation_mode get_compilation_mode() const;

  /**
   * @brief Check if LTO-IR compilation is enabled
   * 
   * @return True if LTO-IR compilation should be attempted
   */
  bool is_lto_ir_enabled() const;

  /**
   * @brief Check if CUDA C++ fallback is allowed
   * 
   * @return True if fallback to CUDA C++ is allowed
   */
  bool is_cuda_fallback_allowed() const;

  /**
   * @brief Set whether to use aggressive LTO-IR operator detection
   * 
   * When enabled, the system will try harder to detect operators that 
   * can be compiled with LTO-IR, potentially at the cost of some accuracy.
   * 
   * @param aggressive True to enable aggressive detection
   */
  void set_aggressive_operator_detection(bool aggressive);

  /**
   * @brief Get whether aggressive operator detection is enabled
   * 
   * @return True if aggressive detection is enabled
   */
  bool is_aggressive_operator_detection_enabled() const;

  /**
   * @brief Load configuration from environment variables
   * 
   * Supported environment variables:
   * - CUDF_JIT_COMPILATION_MODE: "auto", "lto_ir_only", "cuda_only", "prefer_lto_ir"
   * - CUDF_JIT_AGGRESSIVE_DETECTION: "true" or "false"
   */
  void load_from_environment();

 private:
  jit_config();
  
  compilation_mode _mode = compilation_mode::AUTO;
  bool _aggressive_detection = false;
};

}  // namespace jit
}  // namespace cudf