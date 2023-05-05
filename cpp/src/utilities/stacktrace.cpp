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

#include <cudf/detail/stacktrace.hpp>

#include <cxxabi.h>
#include <execinfo.h>
#include <stdlib.h>

#include <sstream>

namespace cudf::detail {

std::string get_stacktrace(capture_last_stackframe capture_last_frame)
{
#ifdef __GNUC__
  constexpr int max_stack_depth = 64;
  void* stack[max_stack_depth];

  auto const depth   = backtrace(stack, max_stack_depth);
  auto const modules = backtrace_symbols(stack, depth);

  if (modules == nullptr) { return "No stacktrace could be captured!"; }

  std::stringstream ss;
  size_t func_name_size = 256;
  char* func_name       = reinterpret_cast<char*>(malloc(func_name_size));

  // Skip one more depth to avoid including the stackframe of this function.
  auto const skip_depth = 1 + (capture_last_frame == capture_last_stackframe::YES ? 0 : 1);
  for (auto i = skip_depth; i < depth; ++i) {
    char* begin_name   = nullptr;
    char* begin_offset = nullptr;
    char* end_offset   = nullptr;

    // Find parentheses and +address offset surrounding the mangled name:
    // ./module(function+0x15c) [0x8048a6d]
    for (char* p = modules[i]; *p; ++p) {
      if (*p == '(') {
        begin_name = p;
      } else if (*p == '+') {
        begin_offset = p;
      } else if (*p == ')' && begin_offset) {
        end_offset = p;
        break;
      }
    }

    auto const frame_idx = i - skip_depth;

    if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
      *begin_name++   = '\0';
      *begin_offset++ = '\0';
      *end_offset     = '\0';

      // Mangled name is now in [begin_name, begin_offset) and caller offset
      // in [begin_offset, end_offset). Apply `__cxa_demangle()`:
      int status{0};
      char* ret = abi::__cxa_demangle(begin_name, func_name, &func_name_size, &status);
      if (status == 0 /*success*/) {
        // __cxa_demangle may realloc func_name.
        func_name = ret;
        ss << "#" << frame_idx << ": " << modules[i] << " : " << func_name << "+" << begin_offset
           << "\n";
      } else {
        // Demangling failed. Output function name as a C function with no arguments.
        ss << "#" << frame_idx << ": " << modules[i] << " : " << begin_name << "()+" << begin_offset
           << "\n";
      }
    } else {
      ss << "#" << frame_idx << ": " << modules[i] << "\n";
    }
  }

  free(func_name);
  free(modules);

  return ss.str();
#else
  return std::string{"Stacktrace is only supported when built with a GNU compiler."};
#endif  // __GNUC__
}

}  // namespace cudf::detail
