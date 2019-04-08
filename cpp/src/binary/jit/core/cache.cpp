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
#include <jitify.hpp>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>

namespace cudf {
namespace jit {

cudfJitCache::cudfJitCache() { }

cudfJitCache::~cudfJitCache() { }

std::shared_ptr<jitify_v2::Program> cudfJitCache::getProgram(
    std::string prog_file_name, 
    std::string const& cuda_source = "",
    std::vector<std::string> const& given_headers = {},
    std::vector<std::string> const& given_options = {},
    jitify_v2::file_callback_type file_callback = nullptr)
{
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(_program_cache_mutex);

    // Find memory cached preprocessed program
    auto prog_it = program_map.find(prog_file_name);
    if ( prog_it != program_map.end()) {
        return prog_it->second;
    }
    else { // Find file cached preprocessed program
        std::string prog_string;
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
        // Add deserialized program to cache and return
        auto program = std::make_shared<jitify_v2::Program>(
            jitify_v2::Program::deserialize(prog_string));
        program_map[prog_file_name] = program;
        return program;
    }
}

// todo: try auto return type
std::shared_ptr<jitify_v2::KernelInstantiation> cudfJitCache::getKernelInstantiation(
    std::string const& kern_name,
    jitify_v2::Program const& program,
    std::vector<std::string> const& arguments)
{
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(_kernel_cache_mutex);

    // Make instance name e.g. "kernel_v_v_int_int_long int_Add"
    std::string kern_inst_name = kern_name;
    for ( auto&& arg : arguments ) kern_inst_name += '_' + arg;

    // Find memory cached kernel instantiation
    auto kern_inst_it = kernel_inst_map.find(kern_inst_name);
    if ( kern_inst_it != kernel_inst_map.end()) {
        return kern_inst_it->second;
    }
    else { // Find file cached kernel instantiation
        std::string kern_inst_string;
        std::ifstream kern_file (kern_inst_name, std::ios::binary);
        if (kern_file) {
            std::stringstream buffer;
            buffer << kern_file.rdbuf();
            kern_inst_string = buffer.str();
        }
        else { // JIT compile the kernel and write to file
            kern_inst_string = program.kernel(kern_name)
                                      .instantiate(arguments)
                                      .serialize();
            std::ofstream kern_file(kern_inst_name, std::ios::binary);
            kern_file << kern_inst_string;
        }
        // Add deserialized kernel to cache and return
        auto kernel = std::make_shared<jitify_v2::KernelInstantiation>(
            jitify_v2::KernelInstantiation::deserialize(kern_inst_string));
        kernel_inst_map[kern_inst_name] = kernel;
        return kernel;
    }
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
