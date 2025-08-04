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

#include "lto_ir.hpp"
#include "config.hpp"

#include <cudf/utilities/error.hpp>

#include <functional>
#include <mutex>
#include <set>

namespace cudf {
namespace jit {

#ifdef CUDF_USE_LTO_IR

lto_ir_cache& lto_ir_cache::instance()
{
  static lto_ir_cache cache;
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    initialize_builtin_lto_ir_operators();
  });
  return cache;
}

std::shared_ptr<lto_ir_cache::lto_ir_operator> lto_ir_cache::get_operator(
  std::string const& operator_name)
{
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _operators.find(operator_name);
  return it != _operators.end() ? it->second : nullptr;
}

void lto_ir_cache::register_operator(std::string const& operator_name,
                                    std::vector<std::string> const& lto_ir_data,
                                    std::vector<std::string> const& dependencies)
{
  std::lock_guard<std::mutex> lock(_mutex);
  auto op = std::make_shared<lto_ir_operator>();
  op->name = operator_name;
  op->data = lto_ir_data;
  op->dependencies = dependencies;
  _operators[operator_name] = op;
}

jitify2::Program lto_ir_cache::link_operators(std::vector<std::string> const& operators,
                                              std::string const& kernel_name)
{
  std::vector<std::string> lto_ir_modules;
  std::set<std::string> processed_deps;
  
  // Collect all operators and their dependencies
  std::function<void(std::string const&)> collect_operator = [&](std::string const& op_name) {
    if (processed_deps.count(op_name)) return;
    
    auto op = get_operator(op_name);
    if (!op) {
      CUDF_FAIL("LTO-IR operator not found: " + op_name);
    }
    
    // Process dependencies first
    for (auto const& dep : op->dependencies) {
      collect_operator(dep);
    }
    
    // Add this operator's LTO-IR data
    lto_ir_modules.insert(lto_ir_modules.end(), op->data.begin(), op->data.end());
    processed_deps.insert(op_name);
  };
  
  for (auto const& op : operators) {
    collect_operator(op);
  }
  
  // Note: jitify2 may not directly support LTO-IR yet. This is a placeholder
  // for when such support becomes available. For now, this would throw.
  // In a real implementation, we'd need either:
  // 1. jitify2 LTO-IR support, or  
  // 2. Direct CUDA driver API calls to load and link LTO-IR
  
  throw std::runtime_error("LTO-IR linking not yet implemented in jitify2");
}

bool lto_ir_cache::is_lto_ir_available(std::string const& operation_type,
                                       std::string const& operator_name) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  std::string full_name = operation_type + "::" + operator_name;
  return _operators.find(full_name) != _operators.end() ||
         _operators.find(operator_name) != _operators.end();
}

void initialize_builtin_lto_ir_operators()
{
#ifdef CUDF_USE_LTO_IR
  // This function will be implemented in the auto-generated lto_ir_data.cpp file
  // when LTO-IR operators are actually generated at build time.
  // For now, register empty placeholder operators to test the infrastructure.
  
  auto& cache = lto_ir_cache::instance();
  
  // Register placeholder operators (empty data for now)
  std::vector<std::string> empty_data;
  
  // Binary arithmetic operators
  cache.register_operator("binary_op::add", empty_data);
  cache.register_operator("binary_op::subtract", empty_data);
  cache.register_operator("binary_op::multiply", empty_data);
  cache.register_operator("binary_op::divide", empty_data);
  
  // Binary comparison operators
  cache.register_operator("binary_op::equal", empty_data);
  cache.register_operator("binary_op::not_equal", empty_data);
  cache.register_operator("binary_op::less", empty_data);
  cache.register_operator("binary_op::greater", empty_data);
  cache.register_operator("binary_op::less_equal", empty_data);
  cache.register_operator("binary_op::greater_equal", empty_data);
  
  // Binary logical operators
  cache.register_operator("binary_op::logical_and", empty_data);
  cache.register_operator("binary_op::logical_or", empty_data);
  
  // Transform operators (common mathematical functions)
  cache.register_operator("transform::sin", empty_data);
  cache.register_operator("transform::cos", empty_data);
  cache.register_operator("transform::exp", empty_data);
  cache.register_operator("transform::log", empty_data);
  cache.register_operator("transform::sqrt", empty_data);
  cache.register_operator("transform::abs", empty_data);
#endif
}

std::unique_ptr<jitify2::Kernel> try_compile_with_lto_ir(
  std::string const& operation_type,
  std::vector<std::string> const& operators,
  std::string const& kernel_name,
  std::string const& fallback_cuda_source)
{
  auto& config = jit_config::instance();
  
  // Check if LTO-IR compilation is enabled
  if (!config.is_lto_ir_enabled()) {
    return nullptr;
  }
  
  auto& cache = lto_ir_cache::instance();
  
  // Check if all required operators are available in LTO-IR
  for (auto const& op : operators) {
    if (!cache.is_lto_ir_available(operation_type, op)) {
      // At least one operator not available
      if (config.get_compilation_mode() == jit_config::compilation_mode::LTO_IR_ONLY) {
        CUDF_FAIL("LTO-IR compilation required but operator not available: " + op);
      }
      // Fall back to CUDA C++ if allowed
      return nullptr;
    }
  }
  
  try {
    // All operators available, attempt LTO-IR compilation
    auto program = cache.link_operators(operators, kernel_name);
    return std::make_unique<jitify2::Kernel>(program.get_kernel(kernel_name));
  } catch (...) {
    // LTO-IR compilation failed
    if (config.get_compilation_mode() == jit_config::compilation_mode::LTO_IR_ONLY) {
      throw; // Re-throw if LTO-IR is required
    }
    // Fall back to CUDA C++ if allowed
    return nullptr;
  }
}

#endif // CUDF_USE_LTO_IR

}  // namespace jit
}  // namespace cudf