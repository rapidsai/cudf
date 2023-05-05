/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cxxabi.h>
#include <execinfo.h>
#include <stdlib.h>

#include <sstream>
#include <string>

namespace cudf {

/**
 * @brief Query the current stack trace and return as string.
 *
 * @param skip_depth The depth to skip from including into the output string
 * @return A string of the current stack trace
 */
std::string get_stacktrace(int skip_depth)
{
#ifdef __GNUC__
  // Adapted from from https://panthema.net/2008/0901-stacktrace-demangled/
  constexpr int kMaxStackDepth = 64;
  void* stack[kMaxStackDepth];

  auto const depth   = backtrace(stack, kMaxStackDepth);
  auto const strings = backtrace_symbols(stack, depth);
  std::stringstream ss;

  if (strings == nullptr) {
    return std::string{"No stack trace could be found!"};
  } else {
    // allocate string which will be filled with the demangled function name
    size_t funcnamesize = 256;
    char* funcname      = (char*)malloc(funcnamesize);

    // Skip one more depth to avoid including the stackframe of this function.
    for (int i = skip_depth + 1; i < depth; ++i) {
      char* begin_name   = nullptr;
      char* begin_offset = nullptr;
      char* end_offset   = nullptr;

      // find parentheses and +address offset surrounding the mangled name:
      // ./module(function+0x15c) [0x8048a6d]
      for (char* p = strings[i]; *p; ++p) {
        if (*p == '(') {
          begin_name = p;
        } else if (*p == '+') {
          begin_offset = p;
        } else if (*p == ')' && begin_offset) {
          end_offset = p;
          break;
        }
      }

      if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
        *begin_name++   = '\0';
        *begin_offset++ = '\0';
        *end_offset     = '\0';

        // mangled name is now in [begin_name, begin_offset) and caller offset
        // in [begin_offset, end_offset). now apply __cxa_demangle():

        int status;
        char* ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
        if (status == 0) {
          funcname = ret;  // use possibly realloc()-ed string (__cxa_demangle may realloc funcname)
          ss << "#" << i << " in " << strings[i] << " : " << funcname << "+" << begin_offset
             << "\n";
        } else {
          // demangling failed. Output function name as a C function with no arguments.
          ss << "#" << i << " in " << strings[i] << " : " << begin_name << "()+" << begin_offset
             << "\n";
        }
      } else {
        ss << "#" << i << " in " << strings[i] << "\n";
      }
    }

    free(funcname);
  }
  free(strings);

  return ss.str();
#else
  return std::string{"Stacktrace is only supported when built with a GNU compiler."};
#endif  // __GNUC__
}

}  // namespace cudf
