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

#ifndef GDF_BINARY_OPERATION_JIT_CORE_LAUNCHER_H
#define GDF_BINARY_OPERATION_JIT_CORE_LAUNCHER_H

#include "../util/type.h"
#include <jit/cache.h>
#include <jit/type.h>
#include <jitify.hpp>
#include <unordered_map>
#include <string>
#include <fstream>
#include <memory>
#include <chrono>

namespace cudf {
namespace rolling {
namespace jit {

    std::istream* headersCode(std::string filename, std::iostream& stream);

    /**---------------------------------------------------------------------------*
     * @brief Class used to handle compilation and execution of JIT kernels
     * 
     *---------------------------------------------------------------------------**/
    class launcher {
    public:
        launcher() = delete;
        
        launcher(const std::string& ptx, const std::string& output_type);

        launcher(launcher&&);

    public:
        launcher(const launcher&) = delete;

        launcher& operator=(launcher&&) = delete;

        launcher& operator=(const launcher&) = delete;

    public:
        /**---------------------------------------------------------------------------*
         * @brief Sets the kernel to launch using this launcher
         * 
         * Method to generate vector containing all template types for a JIT kernel.
         *  This vector is used to get the compiled kernel for one set of types and set
         *  it as the kernel to launch using this launcher.
         * 
         * @tparam Args  Output dtype, LHS dtype, RHS dtype
         * @param type   Operator type (direct (lhs op rhs) or reverse (rhs op lhs))
         * @param args   gdf_column* output, lhs, rhs
         * @return launcher& ref to this launcehr object
         *---------------------------------------------------------------------------**/
        template <typename ... Args>
        launcher& set_kernel_inst(
            std::string&& kernel_name,
            gdf_agg_op ope,
            Args&& ... args)
        {
            std::vector<std::string> arguments;
            arguments.assign({cudf::jit::getTypeName(args.dtype)..., cudf::rolling::jit::get_operator_name(ope)});
            kernel_inst = cache_instance.getKernelInstantiation(kernel_name, program, arguments);
            return *this;
        }

        /**---------------------------------------------------------------------------*
         * @brief Set the Program for this launcher
         * 
         * @param prog_file_name Name to give to the program held by this launcher.
         * 
         * @param ptx Additional ptx code that contains a user defined function to be used.
         * 
         * @return launcher& ref to this launcher object
         *---------------------------------------------------------------------------**/
        launcher& set_program(std::string prog_file_name, std::string ptx, std::string output_type);

        /**---------------------------------------------------------------------------*
          TODO: Update doc.
         * @brief Handle the Jitify API to instantiate and launch using information 
         *  contained in the members of `this`
         * 
         * @param out[out] Output column
         * @param lhs[in]  LHS column
         * @param rhs[in]  RHS column
         * @return gdf_error 
         *---------------------------------------------------------------------------**/
        gdf_error launch(
            gdf_column& out,
            const gdf_column& in,
            gdf_size_type window,
            gdf_size_type min_periods,
            gdf_size_type forward_window,
            const gdf_size_type *window_col,
            const gdf_size_type *min_periods_col,
            const gdf_size_type *forward_window_col
        );

    private:
        static const std::vector<std::string> compiler_flags;
        static const std::vector<std::string> headers_name;

    private:
        cudf::jit::cudfJitCache& cache_instance;
        cudf::jit::named_prog<jitify_v2::Program> program;
        cudf::jit::named_prog<jitify_v2::KernelInstantiation> kernel_inst;

    private:
        jitify_v2::KernelInstantiation& get_kernel() { return *std::get<1>(kernel_inst); }
    };

} // namespace jit
} // namespace rolling
} // namespace cudf

#endif
