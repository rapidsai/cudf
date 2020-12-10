/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include <jit/libcudacxx/cuda/std/chrono.jit>
#include <jit/libcudacxx/cuda/std/climits.jit>
#include <jit/libcudacxx/cuda/std/cstddef.jit>
#include <jit/libcudacxx/cuda/std/cstdint.jit>
#include <jit/libcudacxx/cuda/std/ctime.jit>
#include <jit/libcudacxx/cuda/std/detail/__config.jit>
#include <jit/libcudacxx/cuda/std/detail/__pragma_pop.jit>
#include <jit/libcudacxx/cuda/std/detail/__pragma_push.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/__config.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_pop.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/__pragma_push.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/__undef_macros.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/chrono.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/climits.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/cstddef.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/cstdint.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/ctime.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/limits.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/ratio.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/type_traits.jit>
#include <jit/libcudacxx/cuda/std/detail/libcxx/include/version.jit>
#include <jit/libcudacxx/cuda/std/limits.jit>
#include <jit/libcudacxx/cuda/std/ratio.jit>
#include <jit/libcudacxx/cuda/std/type_traits.jit>
#include <jit/libcudacxx/cuda/std/version.jit>

#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudf {
namespace jit {

const std::vector<std::string> compiler_flags
{
  "-std=c++14",
    // Have jitify prune unused global variables
    "-remove-unused-globals",
    // suppress all NVRTC warnings
    "-w",
    // force libcudacxx to not include system headers
    "-D__CUDACC_RTC__",
#if defined(__powerpc64__)
    "-D__powerpc64__"
#elif defined(__x86_64__)
    "-D__x86_64__"
#endif
};

const std::unordered_map<std::string, char const*> stringified_headers{
  {"cuda/std/chrono", cuda_std_chrono},
  {"cuda/std/climits", cuda_std_climits},
  {"cuda/std/cstddef", cuda_std_cstddef},
  {"cuda/std/cstdint", cuda_std_cstdint},
  {"cuda/std/ctime", cuda_std_ctime},
  {"cuda/std/limits", cuda_std_limits},
  {"cuda/std/ratio", cuda_std_ratio},
  {"cuda/std/type_traits", cuda_std_type_traits},
  {"cuda/std/type_traits", cuda_std_type_traits},
  {"cuda/std/version", cuda_std_version},
  {"cuda/std/detail/__config", cuda_std_detail___config},
  {"cuda/std/detail/__pragma_pop", cuda_std_detail___pragma_pop},
  {"cuda/std/detail/__pragma_push", cuda_std_detail___pragma_push},
  {"cuda/std/detail/libcxx/include/__config", cuda_std_detail_libcxx_include___config},
  {"cuda/std/detail/libcxx/include/__pragma_pop", cuda_std_detail_libcxx_include___pragma_pop},
  {"cuda/std/detail/libcxx/include/__pragma_push", cuda_std_detail_libcxx_include___pragma_push},
  {"cuda/std/detail/libcxx/include/__undef_macros", cuda_std_detail_libcxx_include___undef_macros},
  {"cuda/std/detail/libcxx/include/chrono", cuda_std_detail_libcxx_include_chrono},
  {"cuda/std/detail/libcxx/include/climits", cuda_std_detail_libcxx_include_climits},
  {"cuda/std/detail/libcxx/include/cstddef", cuda_std_detail_libcxx_include_cstddef},
  {"cuda/std/detail/libcxx/include/cstdint", cuda_std_detail_libcxx_include_cstdint},
  {"cuda/std/detail/libcxx/include/ctime", cuda_std_detail_libcxx_include_ctime},
  {"cuda/std/detail/libcxx/include/limits", cuda_std_detail_libcxx_include_limits},
  {"cuda/std/detail/libcxx/include/ratio", cuda_std_detail_libcxx_include_ratio},
  {"cuda/std/detail/libcxx/include/type_traits", cuda_std_detail_libcxx_include_type_traits},
  {"cuda/std/detail/libcxx/include/version", cuda_std_detail_libcxx_include_version},
};

inline std::istream* send_stringified_header(std::iostream& stream, char const* header)
{
  // skip the filename line added by stringify
  stream << (std::strchr(header, '\n') + 1);
  return &stream;
}

}  // namespace jit
}  // namespace cudf
