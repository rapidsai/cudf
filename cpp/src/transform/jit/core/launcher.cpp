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

#include "../core/launcher.h"
#include "../code/code.h"
#include <jit/parser.h>
// #include <jit/types_h_jit.h>
// #include <types.h.jit>
#include <cstdint>
#include <chrono>

extern const char cudf_types_h[];

namespace cudf {
namespace transformation {
namespace jit {

    constexpr char prog_name[] = "unary_transform";

    const std::vector<std::string> Launcher::compilerFlags { "-std=c++14" };
    const std::vector<std::string> Launcher::headersName 
        { cudf_types_h };

    /**---------------------------------------------------------------------------*
     * @brief  Used to provide Jitify with strings that should be used as headers
     *  during JIT compilation.
     * 
     * @param filename  file which was requested to include in source
     * @param stream    stream to pass string of the requested header to
     * @return std::istream* 
     *---------------------------------------------------------------------------**/
    std::istream* headersCode(std::string filename, std::iostream& stream) {
        return nullptr;
    }

    Launcher& Launcher::setProgram(std::string prog_file_name)
    {
        program = cacheInstance.getProgram(prog_file_name,
                                           code::kernel,
                                           headersName,
                                           compilerFlags,
                                           headersCode);
    }
    
    Launcher& Launcher::setProgram(std::string prog_file_name, std::string ptx, std::string output_type)
    {
        std::string combined_kernel = 
          parse_single_function_ptx(ptx, "GENERIC_UNARY_OP", output_type) + code::kernel;
        program = cacheInstance.getProgram(prog_file_name,
                                           combined_kernel.c_str(),
                                           headersName,
                                           compilerFlags,
                                           headersCode);
    }
   
    Launcher::Launcher()
     : cacheInstance{cudf::jit::cudfJitCache::Instance()}
    { 
        this->setProgram(prog_name);
    }

    Launcher::Launcher(const std::string& ptx, const std::string& output_type)
     : cacheInstance{cudf::jit::cudfJitCache::Instance()}
    {
        std::string ptx_hash_str = std::to_string( std::hash<std::string>{}(ptx + output_type) ); 
        this->setProgram(prog_name + ("." + ptx_hash_str), ptx, output_type);
    }
 
    Launcher::Launcher(Launcher&& launcher)
     : program {std::move(launcher.program)}
     , cacheInstance {cudf::jit::cudfJitCache::Instance()}
     , kernel_inst {std::move(launcher.kernel_inst)}
    { }

    gdf_error Launcher::launch(gdf_column* out, gdf_column* in) {

        getKernel().configure_1d_max_occupancy()
                      .launch(out->size,
                              out->data, in->data);

        return GDF_SUCCESS;
    }

} // namespace jit
} // namespace transformation
} // namespace cudf
