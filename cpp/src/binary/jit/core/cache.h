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

#ifndef CUDF_JIT_CACHE_H_
#define CUDF_JIT_CACHE_H_

#include <utilities/error_utils.hpp>
#include <jitify.hpp>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>

namespace cudf {
namespace jit {

std::string getTempDir();

class cudfJitCache
{
public:
    static cudfJitCache& Instance() {
        // Meyers' singleton is thread safe in C++11
        // Link: https://stackoverflow.com/a/1661564
        static cudfJitCache cache;
        return cache;
    }

    cudfJitCache();
    ~cudfJitCache();

    std::shared_ptr<jitify_v2::KernelInstantiation> getKernelInstantiation(
        std::string const& kern_name,
        jitify_v2::Program const& program,
        std::vector<std::string> const& arguments);

    std::shared_ptr<jitify_v2::Program> getProgram(
        std::string prog_file_name, 
        std::string const& cuda_source,
        std::vector<std::string> const& given_headers,
        std::vector<std::string> const& given_options,
        jitify_v2::file_callback_type file_callback);

private:
    template <typename Tv>
    using umap_str_shptr = std::unordered_map<std::string, std::shared_ptr<Tv>>;

    umap_str_shptr<jitify_v2::KernelInstantiation>  kernel_inst_map;
    umap_str_shptr<jitify_v2::Program>              program_map;

    std::mutex _kernel_cache_mutex;
    std::mutex _program_cache_mutex;

private:
    class cacheFile
    {
    private:
        std::string _file_name;
        bool successful_read = false;
        bool successful_write = false;
    public:
        cacheFile(std::string file_name);
        ~cacheFile();

        std::string read();
        void write(std::string);
        bool is_read_successful() { return successful_read; }
        bool is_write_successful() { return successful_write; }
    };

private:
    template <typename T, typename FallbackFunc>
    std::shared_ptr<T> getCached(
        std::string name,
        umap_str_shptr<T>& map,
        FallbackFunc func) {

        // Find memory cached T object
        auto it = map.find(name);
        if ( it != map.end()) {
            return it->second;
        }
        else { // Find file cached T object
            bool successful_read = false;
            std::string serialized;
            #if defined(JITIFY_USE_CACHE)
                std::string file_name = 
                    getTempDir() + name + CUDF_STRINGIFY(CUDF_VERSION);
                cacheFile file{file_name};
                serialized = file.read();
                successful_read = file.is_read_successful();
            #endif
            if (not successful_read) {
                // JIT compile and write to file if possible
                serialized = func().serialize();
                #if defined(JITIFY_USE_CACHE)
                    file.write(serialized);
                #endif
            }
            // Add deserialized T to cache and return
            auto program = std::make_shared<T>(T::deserialize(serialized));
            map[name] = program;
            return program;
        }
    }
};

} // namespace jit
} // namespace cudf


#endif // CUDF_JIT_CACHE_H_