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

std::string getTempDir()
{
    char const *tmpdir_path;
    
    #if defined(__unix__)
        tmpdir_path = getenv("TMPDIR");
        if (tmpdir_path != 0)
            return std::string(tmpdir_path) + '/';
        tmpdir_path = getenv("TMP");
        if (tmpdir_path != 0)
            return std::string(tmpdir_path) + '/';
        tmpdir_path = getenv("TEMP");
        if (tmpdir_path != 0)
            return std::string(tmpdir_path) + '/';
        tmpdir_path = getenv("TEMPDIR");
        if (tmpdir_path != 0)
            return std::string(tmpdir_path) + '/';
        
        tmpdir_path = "/tmp";
        return std::string(tmpdir_path) + '/';
    #elif
        #error Only unix is supported
    #endif // __unix__
}

cudfJitCache::cudfJitCache() { }

cudfJitCache::~cudfJitCache() { }

std::shared_ptr<jitify_v2::Program> cudfJitCache::getProgram(
    std::string prog_name, 
    std::string const& cuda_source = "",
    std::vector<std::string> const& given_headers = {},
    std::vector<std::string> const& given_options = {},
    jitify_v2::file_callback_type file_callback = nullptr)
{
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(_program_cache_mutex);

    return getCached(prog_name, program_map, 
        [&](){return jitify_v2::Program(cuda_source,
                                        given_headers,
                                        given_options,
                                        file_callback);
        }
    );
}

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

    return getCached(kern_inst_name, kernel_inst_map, 
        [&](){return program.kernel(kern_name)
                            .instantiate(arguments);
        }
    );
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
