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

#include <jitify2.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudf {
namespace jit {

#ifdef CUDF_USE_LTO_IR

/**
 * @brief Cache for pre-compiled LTO-IR modules
 * 
 * This class manages a collection of pre-compiled LTO-IR modules that can be
 * linked at runtime to avoid expensive CUDA C++ compilation.
 */
class lto_ir_cache {
 public:
  /**
   * @brief Structure representing a pre-compiled LTO-IR operator
   */
  struct lto_ir_operator {
    std::string name;              ///< Operator name/identifier
    std::vector<std::string> data; ///< LTO-IR binary data
    std::vector<std::string> dependencies; ///< Required dependencies
  };

  /**
   * @brief Get the global LTO-IR cache instance
   */
  static lto_ir_cache& instance();

  /**
   * @brief Load a pre-compiled LTO-IR operator
   * 
   * @param operator_name Name of the operator to load
   * @return The LTO-IR operator if found, nullptr otherwise
   */
  std::shared_ptr<lto_ir_operator> get_operator(std::string const& operator_name);

  /**
   * @brief Register a new LTO-IR operator
   * 
   * @param operator_name Name of the operator
   * @param lto_ir_data LTO-IR binary data
   * @param dependencies List of dependencies
   */
  void register_operator(std::string const& operator_name,
                        std::vector<std::string> const& lto_ir_data,
                        std::vector<std::string> const& dependencies = {});

  /**
   * @brief Link LTO-IR operators into a final program
   * 
   * @param operators List of operators to link
   * @param kernel_name Name of the final kernel
   * @return Jitify program ready for execution
   */
  jitify2::Program link_operators(std::vector<std::string> const& operators,
                                  std::string const& kernel_name);

  /**
   * @brief Check if LTO-IR is available for a given operation
   * 
   * @param operation_type Type of operation (e.g., "binary_op", "transform")
   * @param operator_name Specific operator name
   * @return True if LTO-IR is available, false otherwise
   */
  bool is_lto_ir_available(std::string const& operation_type,
                          std::string const& operator_name) const;

 private:
  lto_ir_cache() = default;
  
  std::unordered_map<std::string, std::shared_ptr<lto_ir_operator>> _operators;
  std::mutex _mutex;
};

/**
 * @brief Initialize pre-compiled LTO-IR operators
 * 
 * This function registers all the built-in LTO-IR operators that were
 * pre-compiled during build time.
 */
void initialize_builtin_lto_ir_operators();

/**
 * @brief Attempt to compile using LTO-IR instead of CUDA C++
 * 
 * @param operation_type Type of operation (e.g., "binary_op", "transform")
 * @param operators List of operators needed for the operation
 * @param kernel_name Name of the kernel to generate
 * @param fallback_cuda_source CUDA C++ source code to use as fallback
 * @return Jitify kernel if successful, or nullptr to indicate fallback needed
 */
std::unique_ptr<jitify2::Kernel> try_compile_with_lto_ir(
  std::string const& operation_type,
  std::vector<std::string> const& operators,
  std::string const& kernel_name,
  std::string const& fallback_cuda_source);

#endif // CUDF_USE_LTO_IR

}  // namespace jit
}  // namespace cudf