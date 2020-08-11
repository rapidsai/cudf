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

#include <iostream>
#include <libcudacxx/details/__config.jit>
#include <libcudacxx/libcxx/include/__config.jit>
#include <libcudacxx/libcxx/include/__undef_macros.jit>
#include <libcudacxx/libcxx/include/cassert.jit>
#include <libcudacxx/libcxx/include/cfloat.jit>
#include <libcudacxx/libcxx/include/chrono.jit>
#include <libcudacxx/libcxx/include/cmath.jit>
#include <libcudacxx/libcxx/include/ctime.jit>
#include <libcudacxx/libcxx/include/limits.jit>
#include <libcudacxx/libcxx/include/ratio.jit>
#include <libcudacxx/libcxx/include/type_traits.jit>
#include <libcudacxx/simt/cassert.jit>
#include <libcudacxx/simt/cfloat.jit>
#include <libcudacxx/simt/chrono.jit>
#include <libcudacxx/simt/cmath.jit>
#include <libcudacxx/simt/ctime.jit>
#include <libcudacxx/simt/limits.jit>
#include <libcudacxx/simt/ratio.jit>
#include <libcudacxx/simt/type_traits.jit>
#include <libcudacxx/simt/version.jit>
#include <string>

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
    // __CHAR_BIT__ is from GCC, but libcxx uses it
    "-D__CHAR_BIT__=" + std::to_string(__CHAR_BIT__),
    // enable temporary workarounds to compile libcudacxx with nvrtc
    "-D_LIBCUDACXX_HAS_NO_CTIME", "-D_LIBCUDACXX_HAS_NO_WCHAR", "-D_LIBCUDACXX_HAS_NO_CFLOAT",
    "-D_LIBCUDACXX_HAS_NO_STDINT", "-D_LIBCUDACXX_HAS_NO_CSTDDEF", "-D_LIBCUDACXX_HAS_NO_CLIMITS",
    "-D_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS",
#if defined(__powerpc64__)
    "-D__powerpc64__"
#elif defined(__x86_64__)
    "-D__x86_64__"
#endif
};

const std::unordered_map<std::string, char const*> stringified_headers{
  {"details/../../libcxx/include/__config", libcxx_config},
  {"../libcxx/include/__undef_macros", libcxx_undef_macros},
  {"simt/../../libcxx/include/cfloat", libcxx_cfloat},
  {"simt/../../libcxx/include/chrono", libcxx_chrono},
  {"simt/../../libcxx/include/ctime", libcxx_ctime},
  {"simt/../../libcxx/include/limits", libcxx_limits},
  {"simt/../../libcxx/include/ratio", libcxx_ratio},
  {"simt/../../libcxx/include/cmath", libcxx_cmath},
  {"simt/../../libcxx/include/cassert", libcxx_cassert},
  {"simt/../../libcxx/include/type_traits", libcxx_type_traits},
  {"simt/../details/__config", libcudacxx_details_config},
  {"simt/cfloat", libcudacxx_simt_cfloat},
  {"simt/chrono", libcudacxx_simt_chrono},
  {"simt/ctime", libcudacxx_simt_ctime},
  {"simt/limits", libcudacxx_simt_limits},
  {"simt/ratio", libcudacxx_simt_ratio},
  {"simt/type_traits", libcudacxx_simt_type_traits},
  {"simt/version", libcudacxx_simt_version},
};

inline std::istream* send_stringified_header(std::iostream& stream, char const* header)
{
  // skip the filename line added by stringify
  stream << (std::strchr(header, '\n') + 1);
  return &stream;
}

}  // namespace jit
}  // namespace cudf
