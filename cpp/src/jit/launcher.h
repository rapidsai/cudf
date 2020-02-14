/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#ifndef GDF_JIT_LAUNCHER_H
#define GDF_JIT_LAUNCHER_H

#include <jit/cache.h>
#include <cudf/types.h>
#include <jitify.hpp>
#include <unordered_map>
#include <string>
#include <fstream>
#include <memory>
#include <chrono>

namespace cudf {
namespace jit {

/**
 * @brief Class used to handle compilation and execution of JIT kernels
 * 
 */
class launcher {
 public:
  launcher() = delete;
   
  /**
   * @brief C'tor of the launcher class
   * 
   * Method to generate vector containing all template types for a JIT kernel.
   *  This vector is used to get the compiled kernel for one set of types and set
   *  it as the kernel to launch using this launcher.
   * 
   * @param hash The hash to be used as the key for caching
   * @param cuda_code The CUDA code that contains the kernel to be launched
   * @param header_names Strings of header_names or strings that contain content
   * of the header files
   * @param compiler_flags Strings of compiler flags
   * @param file_callback a function that returns header file contents given header
   * file names.
   * @param stream The non-owned stream to use for execution
   */
  launcher(
    const std::string& hash,
    const std::string& cuda_source,
    const std::vector<std::string>& header_names,
    const std::vector<std::string>& compiler_flags,
    jitify::experimental::file_callback_type file_callback,
    cudaStream_t stream = 0
  );       
  launcher(launcher&&);
  launcher(const launcher&) = delete;
  launcher& operator=(launcher&&) = delete;
  launcher& operator=(const launcher&) = delete;

  /**
   * @brief Sets the kernel to launch using this launcher
   * 
   * Method to generate vector containing all template types for a JIT kernel.
   *  This vector is used to get the compiled kernel for one set of types and set
   *  it as the kernel to launch using this launcher.
   * 
   * @param kernel_name The kernel to be launched
   * @param arguments   The template arguments to be used to instantiate the kernel
   * @return launcher& ref to this launcehr object
   */
  launcher& set_kernel_inst(
    const std::string& kernel_name,
    const std::vector<std::string>& arguments
  )
  {
    kernel_inst = cache_instance.getKernelInstantiation(kernel_name, program, arguments);
    return *this;
  }

  /**
   * @brief Handle the Jitify API to launch using information 
   *  contained in the members of `this`
   * 
   * @tparam All parameters to launch the kernel
   * @return Return GDF_SUCCESS if successful
   */
  template <typename ... Args>
  void launch(Args ... args){
    get_kernel().configure_1d_max_occupancy(0, 0, 0, stream).launch(args...);
  }

 private:
  cudf::jit::cudfJitCache& cache_instance;
  cudf::jit::named_prog<jitify::experimental::Program> program;
  cudf::jit::named_prog<jitify::experimental::KernelInstantiation> kernel_inst;
  cudaStream_t stream;

  jitify::experimental::KernelInstantiation& get_kernel() { return *std::get<1>(kernel_inst); }
};

} // namespace jit
} // namespace cudf

#endif
