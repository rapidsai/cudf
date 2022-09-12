/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <execinfo.h>
#include <stdlib.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <cstdio>
#include <iostream>

using cudaLaunchKernel_t = cudaError_t (*) (const void*, dim3, dim3, void**, size_t, cudaStream_t);

static cudaLaunchKernel_t cudaLaunchKernel_original; 

void __attribute__((constructor)) init();
void init()
{
    cudaLaunchKernel_original = (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");
}

__host__ cudaError_t cudaLaunchKernel (const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream)
{
    if (stream == static_cast<cudaStream_t>(0)) {
        std::cout << "Found unexpected default stream!" << std::endl;
#ifdef __GNUC__
		constexpr int kMaxStackDepth = 64;
		void* stack[kMaxStackDepth];
		auto depth = backtrace(stack, kMaxStackDepth);
		auto strings = backtrace_symbols(stack, depth);
		if (strings == nullptr) {
			std::cout << "No stack trace could be found!" << std::endl;
		} else {
            // This code is adapted from https://panthema.net/2008/0901-stacktrace-demangled/
            // allocate string which will be filled with the demangled function name
            size_t funcnamesize = 256;
            char *funcname = (char *) malloc(funcnamesize);

		    for (int i = 0; i < depth; ++i) {
                char *begin_name = 0;
                char *begin_offset = 0;
                char *end_offset = 0;

                // find parentheses and +address offset surrounding the mangled name:
                // ./module(function+0x15c) [0x8048a6d]
                for (char *p = strings[i]; *p; ++p)
                {
                    if (*p == '(') {
                      begin_name = p;
                    } else if (*p == '+') {
                        begin_offset = p;
                    } else if (*p == ')' && begin_offset) {
                        end_offset = p;
                        break;
                    }
                }

                if (begin_name && begin_offset && end_offset && begin_name < begin_offset)
                {
                    *begin_name++ = '\0';
                    *begin_offset++ = '\0';
                    *end_offset = '\0';

                    // mangled name is now in [begin_name, begin_offset) and caller
                    // offset in [begin_offset, end_offset). now apply
                    // __cxa_demangle():

                    int status;
                    char* ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
                    if (status == 0) {
                        funcname = ret; // use possibly realloc()-ed string
                        std::cout << "#" << i << " in " << strings[i] << " : " << funcname << "+" << begin_offset << std::endl;
                    } else {
                    // demangling failed. Output function name as a C function with
                    // no arguments.
                        std::cout << "#" << i << " in " << strings[i] << " : " << begin_name << "()+" << begin_offset << std::endl;
                    }
                } else {
                    std::cout << "#" << i << " in " << strings[i] << std::endl;
                }
		    }

            free(funcname);
        }
		free(strings);
#else
        std::cout << "Backtraces are only support on GNU systems." << std::endl;
#endif  // __GNUC__
        std::cout << std::endl;
    }
    return cudaLaunchKernel_original(func, gridDim, blockDim, args, sharedMem, stream);
}
