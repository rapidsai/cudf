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

#include "cache.h"
#include "../util/type.h"
#include "../util/operator.h"
#include <jitify.hpp>
#include <unordered_map>
#include <string>
#include <fstream>
#include <memory>
#include <chrono>

namespace cudf {
namespace binops {
namespace jit {

    std::istream* headersCode(std::string filename, std::iostream& stream);

    /**---------------------------------------------------------------------------*
     * @brief Class used to handle compilation and execution of JIT kernels
     * 
     *---------------------------------------------------------------------------**/
    class Launcher {
    public:
        Launcher();
        
        Launcher(const std::string& ptx);

        Launcher(Launcher&&);

    public:
        Launcher(const Launcher&) = delete;

        Launcher& operator=(Launcher&&) = delete;

        Launcher& operator=(const Launcher&) = delete;

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
         * @return Launcher& ref to this launcehr object
         *---------------------------------------------------------------------------**/
        template <typename ... Args>
        Launcher& setKernelInst(
            std::string&& kernName,
            gdf_binary_operator ope,
            Operator::Type type,
            Args&& ... args)
        {
            Operator operatorSelector;
            std::vector<std::string> arguments;
            arguments.assign({getTypeName(args->dtype)..., operatorSelector.getOperatorName(ope, type)});
            kernel_inst = cacheInstance.getKernelInstantiation(kernName, program, arguments);
            return *this;
        }

        /**---------------------------------------------------------------------------*
         * @brief Set the Program for this launcher
         * 
         * @param prog_file_name Name to give to the program held by this Launcher.
         * @return Launcher& ref to this launcher object
         *---------------------------------------------------------------------------**/
        Launcher& setProgram(std::string prog_file_name);
 
        /**---------------------------------------------------------------------------*
         * @brief Set the Program for this launcher
         * 
         * @param prog_file_name Name to give to the program held by this Launcher.
         * 
         * @param ptx Additional ptx code that contains a user defined function to be used.
         * 
         * @return Launcher& ref to this launcher object
         *---------------------------------------------------------------------------**/
        Launcher& setProgram(std::string prog_file_name, std::string ptx);

        /**---------------------------------------------------------------------------*
         * @brief Handle the Jitify API to instantiate and launch using information 
         *  contained in the members of `this`
         * 
         * @param out[out] Output column
         * @param lhs[in]  LHS column
         * @param rhs[in]  RHS scalar (single value)
         * @return gdf_error 
         *---------------------------------------------------------------------------**/
        gdf_error launch(gdf_column* out, gdf_column* lhs, gdf_scalar* rhs);

        /**---------------------------------------------------------------------------*
         * @brief Handle the Jitify API to instantiate and launch using information 
         *  contained in the members of `this`
         * 
         * @param out[out] Output column
         * @param lhs[in]  LHS column
         * @param rhs[in]  RHS column
         * @return gdf_error 
         *---------------------------------------------------------------------------**/
        gdf_error launch(gdf_column* out, gdf_column* lhs, gdf_column* rhs);

    private:
        static const std::vector<std::string> compilerFlags;
        static const std::vector<std::string> headersName;

    private:
        cudf::jit::cudfJitCache& cacheInstance;
        cudf::jit::named_prog<jitify_v2::Program> program;
        cudf::jit::named_prog<jitify_v2::KernelInstantiation> kernel_inst;

    private:
        jitify_v2::KernelInstantiation& getKernel() { return *std::get<1>(kernel_inst); }
    };

} // namespace jit
} // namespace binops
} // namespace cudf

#endif
