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

#include "cache.h"

namespace cudf {
namespace jit {

cudfJitCache::cudfJitCache() { }

cudfJitCache::~cudfJitCache() { }

/**---------------------------------------------------------------------------*
 * @brief Get 
 * 
 * @param prog_file_name 
 * @param cuda_source 
 * @param given_headers 
 * @param given_options 
 * @param file_callback 
 * @return jitify_v2::Program 
 *---------------------------------------------------------------------------**/
jitify_v2::Program cudfJitCache::getProgram(
    std::string prog_file_name, 
    std::string const& cuda_source = "",
    std::vector<std::string> const& given_headers = {},
    std::vector<std::string> const& given_options = {},
    jitify_v2::file_callback_type file_callback = nullptr)
{
    std::string prog_string;

    // Find memory cached preprocessed program
    auto prog_it = program_map.find(prog_file_name);
    if ( prog_it != program_map.end()) {
        prog_string = prog_it->second;
    }
    else { // Find file cached preprocessed program
        std::ifstream prog_file (prog_file_name, std::ios::binary);
        if (prog_file) {
            std::stringstream buffer;
            buffer << prog_file.rdbuf();
            prog_string = buffer.str();
        }
        else { // JIT preprocess the program and write to file
            prog_string = jitify_v2::Program(cuda_source,
                                            given_headers,
                                            given_options,
                                            file_callback)
                                            .serialize();
            std::ofstream prog_file(prog_file_name, std::ios::binary);
            prog_file << prog_string;
        }
    }

    return jitify_v2::Program::deserialize(prog_string);
}

jitify_v2::KernelInstantiation cudfJitCache::getKernelInstantiation(
    std::string const& kern_name,
    jitify_v2::Program const& program,
    std::vector<std::string> const& arguments)
{
    std::string kern_inst_string;

    // Make instance name e.g. "kernel_v_v_int_int_long int_Add"
    std::string kern_inst_name = kern_name;
    for ( auto&& arg : arguments ) kern_inst_name += '_' + arg;

    // Find memory cached kernel instantiation
    auto kern_inst_it = kernel_inst_map.find(kern_inst_name);
    if ( kern_inst_it != kernel_inst_map.end()) {
        kern_inst_string = kern_inst_it->second;
    }
    else { // Find file cached kernel instantiation
        std::ifstream kern_file (kern_inst_name, std::ios::binary);
        if (kern_file) {
            std::stringstream buffer;
            buffer << kern_file.rdbuf();
            kern_inst_string = buffer.str();
            kernel_inst_map[kern_inst_name] = kern_inst_string;
        }
        else { // JIT compile the kernel and write to file
            kern_inst_string = program.kernel(kern_name)
                                      .instantiate(arguments)
                                      .serialize();
            std::ofstream kern_file(kern_inst_name, std::ios::binary);
            kern_file << kern_inst_string;
            kernel_inst_map[kern_inst_name] = kern_inst_string;
        }
    }

    return jitify_v2::KernelInstantiation::deserialize(kern_inst_string);
}

// Another overload for getKernelInstantiation which might be useful to get
// kernel instantiations in one step
// ------------------------------------------------------------------------
/*
jitify_v2::KernelInstantiation cudfJitCache::getKernelInstantiation(
    std::string const& kern_name,
    std::string const& prog_name,
    std::string const& cuda_source = "",
    std::vector<std::string> const& given_headers = {},
    std::vector<std::string> const& given_options = {},
    file_callback_type file_callback = nullptr)
{
    auto program = getProgram(prog_name,
                              cuda_source,
                              given_headers,
                              given_options,
                              file_callback);
    return getKernelInstantiation(kern_name, program);
}
*/

} // namespace jit
} // namespace cudf
