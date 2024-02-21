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

#include <cudf/detail/utilities/stacktrace.hpp>

#if defined(__GNUC__) && defined(CUDF_BUILD_STACKTRACE_DEBUG)
#include <cxxabi.h>
#include <execinfo.h>

#include <cstdlib>
#include <cstring>
#include <sstream>
#endif  // defined(__GNUC__) && defined(CUDF_BUILD_STACKTRACE_DEBUG)

namespace cudf::detail {

std::string get_stacktrace(capture_last_stackframe capture_last_frame)
{
#if defined(__GNUC__) && defined(CUDF_BUILD_STACKTRACE_DEBUG)
  constexpr int max_stack_depth = 64;
  void* stack[max_stack_depth];

  auto const depth   = backtrace(stack, max_stack_depth);
  auto const modules = backtrace_symbols(stack, depth);

  if (modules == nullptr) { return "No stacktrace could be captured!"; }

  std::stringstream ss;

  // Skip one more depth to avoid including the stackframe of this function.
  auto const skip_depth = 1 + (capture_last_frame == capture_last_stackframe::YES ? 0 : 1);
  for (auto i = skip_depth; i < depth; ++i) {
    // Each modules[i] string contains a mangled name in the format like following:
    // `module_name(function_name+0x012) [0x01234567890a]`
    // We need to extract function name and function offset.
    char* begin_func_name   = std::strstr(modules[i], "(");
    char* begin_func_offset = std::strstr(modules[i], "+");
    char* end_func_offset   = std::strstr(modules[i], ")");

    auto const frame_idx = i - skip_depth;
    if (begin_func_name && begin_func_offset && end_func_offset &&
        begin_func_name < begin_func_offset) {
      // Split `modules[i]` into separate null-terminated strings.
      // After this, mangled function name will then be [begin_func_name, begin_func_offset), and
      // function offset is in [begin_func_offset, end_func_offset).
      *(begin_func_name++)   = '\0';
      *(begin_func_offset++) = '\0';
      *end_func_offset       = '\0';

      // We need to demangle function name.
      int status{0};
      char* func_name = abi::__cxa_demangle(begin_func_name, nullptr, nullptr, &status);

      ss << "#" << frame_idx << ": " << modules[i] << " : "
         << (status == 0 /*demangle success*/ ? func_name : begin_func_name) << "+"
         << begin_func_offset << "\n";
      free(func_name);
    } else {
      ss << "#" << frame_idx << ": " << modules[i] << "\n";
    }
  }

  free(modules);

  return ss.str();
#else
#ifdef CUDF_BUILD_STACKTRACE_DEBUG
  return "Stacktrace is only supported when built with a GNU compiler.";
#else
  return "libcudf was not built with stacktrace support.";
#endif  // CUDF_BUILD_STACKTRACE_DEBUG
#endif  // __GNUC__
}

}  // namespace cudf::detail
