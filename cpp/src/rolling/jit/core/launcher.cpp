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

#include "../code/code.h"
#include "launcher.h"
#include <jit/parser.h>
#include <jit/types_h_jit.h>
#include <cstdint>
#include <chrono>

namespace cudf {
namespace rolling {
namespace jit {

    const std::vector<std::string> launcher::compiler_flags { "-std=c++14" };
    const std::vector<std::string> launcher::headers_name 
        { "operation.h" , cudf_types_h };

    /**---------------------------------------------------------------------------*
     * @brief  Used to provide Jitify with strings that should be used as headers
     *  during JIT compilation.
     * 
     * @param filename  file which was requested to include in source
     * @param stream    stream to pass string of the requested header to
     * @return std::istream* 
     *---------------------------------------------------------------------------**/
    std::istream* headers_code(std::string filename, std::iostream& stream) {
        if (filename == "operation.h") {
            stream << code::operation;
            return &stream;
        }
        return nullptr;
    }

    launcher& launcher::set_program(std::string prog_file_name, std::string ptx, std::string output_type)
    {
        std::string combined_kernel = 
          cudf::jit::parse_single_function_ptx(ptx, "NUMBA_GENERIC_AGGREGATOR", output_type, {0,5}) + code::kernel;
        program = cache_instance.getProgram(prog_file_name,
                                           combined_kernel.c_str(),
                                           headers_name,
                                           compiler_flags,
                                           headers_code);
    }

    launcher::launcher(const std::string& ptx, const std::string& output_type)
     : cache_instance{cudf::jit::cudfJitCache::Instance()}
    {
        std::string ptx_hash_str = std::to_string( std::hash<std::string>{}(ptx + output_type) ); 
        this->set_program("prog_rolling." + ptx_hash_str, ptx, output_type);
    }
 
    launcher::launcher(launcher&& launcher)
     : program {std::move(launcher.program)}
     , cache_instance {cudf::jit::cudfJitCache::Instance()}
     , kernel_inst {std::move(launcher.kernel_inst)}
    { }

    gdf_error launcher::launch(
        gdf_column& out, 
        const gdf_column& in,
        gdf_size_type window,
        gdf_size_type min_periods,
        gdf_size_type forward_window,
        const gdf_size_type *window_col,
        const gdf_size_type *min_periods_col,
        const gdf_size_type *forward_window_col
    ) {
       
        get_kernel().configure_1d_max_occupancy()
                      .launch(
                          out.size,
                          out.data, 
                          out.valid,
                          in.data,
                          in.valid,
                          window,
                          min_periods,
                          forward_window,
                          window_col,
                          min_periods_col,
                          forward_window_col
                        );

        return GDF_SUCCESS;
    }

} // namespace jit
} // namespace rolling
} // namespace cudf
