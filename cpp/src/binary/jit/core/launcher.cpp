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

#include "binary/jit/core/launcher.h"
#include "binary/jit/code/code.h"
#include "types.h.jit"
#include <cstdint>
#include <chrono>

namespace cudf {
namespace binops {
namespace jit {

    const std::vector<std::string> Launcher::compilerFlags { "-std=c++14" };
    const std::vector<std::string> Launcher::headersName 
        { "operation.h" , "traits.h" , cudf_types_h };

    /**---------------------------------------------------------------------------*
     * @brief  Used to provide Jitify with strings that should be used as headers
     *  during JIT compilation.
     * 
     * @param filename  file which was requested to include in source
     * @param stream    stream to pass string of the requested header to
     * @return std::istream* 
     *---------------------------------------------------------------------------**/
    std::istream* headersCode(std::string filename, std::iostream& stream) {
        if (filename == "operation.h") {
            stream << code::operation;
            return &stream;
        }
        if (filename == "traits.h") {
            stream << code::traits;
            return &stream;
        }
        return nullptr;
    }

    jitify_v2::Program Launcher::getProgram(std::string prog_file_name) {
        std::ifstream prog_file (prog_file_name, std::ios::binary);

        // Find file cached preprocessed program
        if (prog_file) {
            std::stringstream buffer;
            buffer << prog_file.rdbuf();
            return jitify_v2::Program::deserialize(buffer.str());
        }
        // JIT preprocess the program and write to file
        else {
            auto _program = jitify_v2::Program(code::kernel,
                                            headersName,
                                            compilerFlags,
                                            headersCode);
            std::ofstream prog_file(prog_file_name, std::ios::binary);
            prog_file << _program.serialize();
            return _program;
        }
    }

    Launcher::Launcher()
     : program {getProgram("prog_binop.jit")}
    { }

    Launcher::Launcher(Launcher&& launcher)
     : program {std::move(launcher.program)}
    { }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* lhs, gdf_scalar* rhs) {
        // program.kernel(kernelName.c_str())
        //        .instantiate(arguments)
        //        .configure_1d_max_occupancy()
        //        .launch(out->size,
        //                out->data, lhs->data, rhs->data);

        return GDF_SUCCESS;
    }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* lhs, gdf_column* rhs) {
        auto startPointClock = std::chrono::high_resolution_clock::now();

        auto kernel_inst = 
            jitify_v2::KernelInstantiation::deserialize(kern_inst_string);
        auto stopPointClock = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = stopPointClock-startPointClock;
        std::cout << "Deserialize (ms): " << elapsed_seconds.count()*1000 << std::endl;

        kernel_inst.configure_1d_max_occupancy()
               .launch(out->size,
                       out->data, lhs->data, rhs->data);

        return GDF_SUCCESS;
    }

} // namespace jit
} // namespace binops
} // namespace cudf
