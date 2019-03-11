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

#include "binary/jit/util/type.h"
#include "binary/jit/util/operator.h"
#include <jitify.hpp>

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
        static Launcher launch() {
            return Launcher();
        }

    public:
        Launcher();

        Launcher(Launcher&&);

    public:
        Launcher(const Launcher&) = delete;

        Launcher& operator=(Launcher&&) = delete;

        Launcher& operator=(const Launcher&) = delete;

    public:
        /**---------------------------------------------------------------------------*
         * @brief  Set the kernel name that this launcher will compile and launch
         * 
         * @param value  kernel name
         *---------------------------------------------------------------------------**/
        Launcher& kernel(std::string&& value);

        /**---------------------------------------------------------------------------*
         * @brief Method to generate vector containing all template types for a JIT
         *  kernel. This vector is used to instantiate the kernel code for one set of types
         * 
         * @tparam Args  Output dtype, LHS dtype, RHS dtype
         * @param type   Operator type (direct (lhs op rhs) or reverse (rhs op lhs))
         * @param args   gdf_column* output, lhs, rhs
         * @return Launcher& ref to this launcehr object
         *---------------------------------------------------------------------------**/
        template <typename ... Args>
        Launcher& instantiate(gdf_binary_operator ope, Operator::Type type, Args&& ... args) {
            Operator operatorSelector;
            arguments.assign({getTypeName(args->dtype)..., operatorSelector.getOperatorName(ope, type)});
            return *this;
        }

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
        jitify::Program program;

    private:
        std::string kernelName;
        std::vector<std::string> arguments;
    };

} // namespace jit
} // namespace binops
} // namespace cudf

#endif
