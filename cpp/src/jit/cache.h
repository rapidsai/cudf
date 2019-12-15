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

#include <jitify.hpp>
#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>
#include <boost/filesystem.hpp>

namespace cudf {
namespace jit {

template <typename Tv>
using named_prog = std::pair<std::string, std::shared_ptr<Tv>>;

/**
 * @brief Get the string path to the JITIFY kernel cache directory.
 *
 * This path can be overridden at runtime by defining an environment variable
 * named `LIBCUDF_KERNEL_CACHE_PATH`. The value of this variable must be a path
 * under which the process' user has read/write priveleges.
 *
 * This function returns a path to the cache directory, creating it if it
 * doesn't exist.
 *
 * The default cache directory `$TEMPDIR/cudf_$CUDF_VERSION`.
 **/
boost::filesystem::path getCacheDir();

class cudfJitCache
{
public:

    /**---------------------------------------------------------------------------*
     * @brief Get a process wide singleton cache object
     * 
     *---------------------------------------------------------------------------**/
    static cudfJitCache& Instance() {
        // Meyers' singleton is thread safe in C++11
        // Link: https://stackoverflow.com/a/1661564
        static cudfJitCache cache;
        return cache;
    }

    cudfJitCache();
    ~cudfJitCache();

    /**---------------------------------------------------------------------------*
     * @brief Get the Kernel Instantiation object
     * 
     * Searches an internal in-memory cache and file based cache for the kernel
     * and if not found, JIT compiles and returns the kernel
     * 
     * @param kern_name [in] name of kernel to return
     * @param program   [in] Jitify preprocessed program to get the kernel from
     * @param arguments [in] template arguments for kernel in vector of strings
     * @return  Pair of string kernel identifier and compiled kernel object
     *---------------------------------------------------------------------------**/
    named_prog<jitify_v2::KernelInstantiation> getKernelInstantiation(
        std::string const& kern_name,
        named_prog<jitify_v2::Program> const& program,
        std::vector<std::string> const& arguments);

    /**---------------------------------------------------------------------------*
     * @brief Get the Jitify preprocessed Program object
     * 
     * Searches an internal in-memory cache and file based cache for the Jitify
     * pre-processed program and if not found, JIT processes and returns it
     * 
     * @param prog_file_name [in] name of program to return
     * @param cuda_source    [in] string source code of program to compile
     * @param given_headers  [in] vector of strings representing source or names of
     *  each header included in cuda_source
     * @param given_options  [in] vector of strings options to pass to NVRTC
     * @param file_callback  [in] pointer to callback function to call whenever a
     *  header needs to be loaded
     * @return named_prog<jitify_v2::Program> 
     *---------------------------------------------------------------------------**/
    named_prog<jitify_v2::Program> getProgram(
        std::string const& prog_file_name, 
        std::string const& cuda_source = "",
        std::vector<std::string> const& given_headers = {},
        std::vector<std::string> const& given_options = {},
        jitify_v2::file_callback_type file_callback = nullptr);

private:
    template <typename Tv>
    using umap_str_shptr = std::unordered_map<std::string, std::shared_ptr<Tv>>;

    umap_str_shptr<jitify_v2::KernelInstantiation>  kernel_inst_map;
    umap_str_shptr<jitify_v2::Program>              program_map;

    /*
    Even though this class can be used as a non-singleton, the file cache
    access should remain limited to one thread per process. The lockf locks can
    prevent multiple processes from accessing the file but are ineffective in
    preventing multiple threads from doing so as the lock is shared by the
    entire process.
    Therefore the mutexes are static.
    */
    static std::mutex _kernel_cache_mutex;
    static std::mutex _program_cache_mutex;

private:
    /**---------------------------------------------------------------------------*
     * @brief Class to allow process wise exclusive access to cache files
     * 
     *---------------------------------------------------------------------------**/
    class cacheFile
    {
    private:
        std::string _file_name;
        bool successful_read = false;
        bool successful_write = false;
    public:
        cacheFile(std::string file_name);
        ~cacheFile();

        /**---------------------------------------------------------------------------*
         * @brief Read this file and return the contents as a std::string
         * 
         *---------------------------------------------------------------------------**/
        std::string read();

        /**---------------------------------------------------------------------------*
         * @brief Write the passed string to this file
         * 
         *---------------------------------------------------------------------------**/
        void write(std::string);

        /**---------------------------------------------------------------------------*
         * @brief Check whether the read() operation on the file completed successfully
         * 
         * @return true Read was successful. String returned by `read()` is valid
         * @return false Read was unsuccessful. String returned by `read()` is empty
         *---------------------------------------------------------------------------**/
        bool is_read_successful() { return successful_read; }

        /**---------------------------------------------------------------------------*
         * @brief Check whether the write() operation on the file completed successfully
         * 
         * @return true Write was successful.
         * @return false Write was unsuccessful. File state is undefined
         *---------------------------------------------------------------------------**/
        bool is_write_successful() { return successful_write; }
    };

private:
    template <typename T, typename FallbackFunc>
    named_prog<T> getCached(
        std::string const& name,
        umap_str_shptr<T>& map,
        FallbackFunc func) {

        // Find memory cached T object
        auto it = map.find(name);
        if ( it != map.end()) {
            return std::make_pair(name, it->second);
        }
        else { // Find file cached T object
            bool successful_read = false;
            std::string serialized;
            #if defined(JITIFY_USE_CACHE)
                boost::filesystem::path file_name = getCacheDir() / name;
                cacheFile file{file_name.string()};
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
            return std::make_pair(name, program);
        }
    }
};

} // namespace jit
} // namespace cudf


#endif // CUDF_JIT_CACHE_H_