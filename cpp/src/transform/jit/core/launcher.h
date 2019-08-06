/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#ifndef GDF_UNARY_TRANSFORM_JIT_CORE_LAUNCHER_H
#define GDF_UNARY_TRANSFORM_JIT_CORE_LAUNCHER_H

#include <jit/cache.h>
#include <jit/type.h>
#include <chrono>
#include <fstream>
#include <jitify.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace cudf {
namespace transformation {
namespace jit {

std::istream* headersCode(std::string filename, std::iostream& stream);

/**
 * @brief Class used to handle compilation and execution of JIT kernels
 *
 **/
class Launcher {
 public:
  Launcher();

  Launcher(const std::string& udf, const std::string& output_type, bool is_ptx);

  Launcher(Launcher&&);

 public:
  Launcher(const Launcher&) = delete;

  Launcher& operator=(Launcher&&) = delete;

  Launcher& operator=(const Launcher&) = delete;

 public:
  /**
   * @brief Sets the kernel to launch using this launcher
   *
   * Method to generate vector containing all template types for a JIT kernel.
   *  This vector is used to get the compiled kernel for one set of types and
   *set it as the kernel to launch using this launcher.
   *
   * @tparam Args  Output dtype, LHS dtype, RHS dtype
   * @param type   Operator type (direct (lhs op rhs) or reverse (rhs op lhs))
   * @param args   gdf_column* output, lhs, rhs
   * @return Launcher& ref to this launcehr object
   **/
  template <typename... Args>
  Launcher& setKernelInst(std::string&& kernName, Args&&... args) {
    std::vector<std::string> arguments;
    arguments.assign({cudf::jit::getTypeName(args->dtype)...});
    kernel_inst =
        cacheInstance.getKernelInstantiation(kernName, program, arguments);
    return *this;
  }

  /**
   * @brief Set the Program for this launcher
   *
   * @param prog_file_name Name to give to the program held by this Launcher.
   *
   * @param ptx Additional ptx code that contains a user defined function to be
   *used.
   *
   * @param output_type The output type that is compatible with the PTX code
   *
   * @return Launcher& ref to this launcher object
   **/
  Launcher& setProgram(std::string prog_file_name, std::string ptx,
                       std::string output_type, bool is_ptx);

  /**
   * @brief Handle the Jitify API to instantiate and launch using information
   *  contained in the members of `this`
   *
   * @param out[out] Output column
   * @param lhs[in]  LHS column
   * @param rhs[in]  RHS scalar (single value)
   * @return gdf_error
   **/
  gdf_error launch(gdf_column* out, const gdf_column* in);

 private:
  static const std::vector<std::string> compilerFlags;
  static const std::vector<std::string> headersName;

 private:
  cudf::jit::cudfJitCache& cacheInstance;
  cudf::jit::named_prog<jitify_v2::Program> program;
  cudf::jit::named_prog<jitify_v2::KernelInstantiation> kernel_inst;

 private:
  jitify_v2::KernelInstantiation& getKernel() {
    return *std::get<1>(kernel_inst);
  }
};

}  // namespace jit
}  // namespace transformation
}  // namespace cudf

#endif
