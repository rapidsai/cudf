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

#include <utilities/error_utils.hpp>
#include "cache.h"
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>

namespace cudf {
namespace jit {

std::string getCacheDir()
{
    char const *tmpdir_path;
    std::string cache_dir{"cudf"};
    cache_dir = cache_dir + '_' + CUDF_STRINGIFY(CUDF_VERSION) + '/';
    
    #if defined(__unix__)

        (tmpdir_path = std::getenv("TMPDIR" )) ||
        (tmpdir_path = std::getenv("TMP"    )) ||
        (tmpdir_path = std::getenv("TEMP"   )) ||
        (tmpdir_path = std::getenv("TEMPDIR"));

        tmpdir_path = ( tmpdir_path != 0 ) ? tmpdir_path : "/tmp";
        
        cache_dir = std::string(tmpdir_path) + '/' + cache_dir;

        // if it doesn't exist, make it
        mkdir(cache_dir.c_str(), S_IRWXU);

        return cache_dir;
    #elif
        #error Only unix is supported
    #endif // __unix__
}

cudfJitCache::cudfJitCache() { }

cudfJitCache::~cudfJitCache() { }

std::mutex cudfJitCache::_kernel_cache_mutex;
std::mutex cudfJitCache::_program_cache_mutex;

named_prog<jitify_v2::Program> cudfJitCache::getProgram(
    std::string const& prog_name, 
    std::string const& cuda_source,
    std::vector<std::string> const& given_headers,
    std::vector<std::string> const& given_options,
    jitify_v2::file_callback_type file_callback)
{
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(_program_cache_mutex);

    return getCached(prog_name, program_map, 
        [&](){
            CUDF_EXPECTS( not cuda_source.empty(),
                "Program not found in cache, Needs source string.");
            return jitify_v2::Program(cuda_source,
                                        given_headers,
                                        given_options,
                                        file_callback);
        }
    );
}

named_prog<jitify_v2::KernelInstantiation> cudfJitCache::getKernelInstantiation(
    std::string const& kern_name,
    named_prog<jitify_v2::Program> const& named_program,
    std::vector<std::string> const& arguments)
{
    // Lock for thread safety
    std::lock_guard<std::mutex> lock(_kernel_cache_mutex);

    std::string prog_name = std::get<0>(named_program);
    jitify_v2::Program& program = *std::get<1>(named_program);

    // Make instance name e.g. "prog_binop.kernel_v_v_int_int_long int_Add"
    std::string kern_inst_name = prog_name + '.' + kern_name;
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

cudfJitCache::cacheFile::cacheFile(std::string file_name)
 : _file_name{file_name}
{ }

cudfJitCache::cacheFile::~cacheFile() { }

std::string cudfJitCache::cacheFile::read()
{
    // Open file (duh)
    int fd = open ( _file_name.c_str(), O_RDWR );
    if ( fd == -1 ) {
        successful_read = false;
        return std::string();
    }

    // Lock the file descriptor. we the only ones now
    if ( lockf(fd, F_LOCK, 0) == -1 ) {
        successful_read = false;
        return std::string();
    }

    // Get file descriptor from file pointer
    FILE *fp = fdopen( fd, "rb" );

    // Get file length
    fseek( fp , 0L , SEEK_END);
    size_t file_size = ftell( fp );
    rewind( fp );

    // Allocate memory of file length size
    std::string content;
    content.resize(file_size);
    char *buffer = &content[0];

    // Copy file into buffer
    if( fread(buffer, file_size, 1, fp) != 1 ) {
        successful_read = false;
        fclose(fp);
        free(buffer);
        return std::string();
    }
    fclose(fp);
    successful_read = true;

    return content;
}

void cudfJitCache::cacheFile::write(std::string content)
{
    // Open file and create if it doesn't exist, with access 0600
    int fd = open ( _file_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR );
    if ( fd == -1 ) {
        successful_write = false;
        return;
    }

    // Lock the file descriptor. we the only ones now
    if ( lockf(fd, F_LOCK, 0) == -1 ) {
        successful_write = false;
        return;
    }

    // Get file descriptor from file pointer
    FILE *fp = fdopen( fd, "wb" );

    // Copy string into file
    if( fwrite(content.c_str(), content.length(), 1, fp) != 1 ) {
        successful_write = false;
        fclose(fp);
        return;
    }
    fclose(fp);

    successful_write = true;
    return;
}

} // namespace jit
} // namespace cudf
