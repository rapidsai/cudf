#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

set(CUDF_MIN_VERSION_libcudacxx 1.4.0)

CPMFindPackage(NAME     libcudacxx
    VERSION             ${CUDF_MIN_VERSION_libcudacxx}
    GIT_REPOSITORY      https://github.com/NVIDIA/libcudacxx.git
    GIT_TAG             ${CUDF_MIN_VERSION_libcudacxx}
    GIT_SHALLOW         TRUE
    DONWLOAD_ONLY       TRUE
    OPTIONS             "LIBCXX_CONFIGURE_IDE OFF"
                        "LIBCXX_ENABLE_STATIC OFF"
                        "LIBCXX_ENABLE_SHARED OFF"
                        "LIBCXX_INCLUDE_TESTS OFF"
                        "LIBCXX_INSTALL_LIBRARY OFF"
                        "LIBCXX_INSTALL_HEADERS OFF"
                        "LIBCXX_STANDALONE_BUILD OFF"
                        "LIBCXX_DISABLE_ARCH_BY_DEFAULT ON"
                        "LIBCXX_INSTALL_SUPPORT_HEADERS OFF"
                        "LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY OFF"
                        # Set this to a place LLVM definitely isn't, to defeat libcu++'s CMakeLists.txt install
                        # targets. Is this a CPM bug? https://github.com/TheLartians/CPM.cmake/issues/173
                        "LLVM_PATH /tmp"
)

message(STATUS "libcudacxx_ADDED: ${libcudacxx_ADDED}")
message(STATUS "libcudacxx_FOUND: ${libcudacxx_FOUND}")

if(NOT (libcudacxx_ADDED OR libcudacxx_FOUND))
    message(FATAL_ERROR "libcudacxx package not found")
endif()

message(STATUS "libcudacxx_SOURCE_DIR: ${libcudacxx_SOURCE_DIR}")

set(LIBCUDACXX_DIR "${libcudacxx_SOURCE_DIR}")
set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include")

message(STATUS "LIBCUDACXX_DIR: ${LIBCUDACXX_DIR}")
message(STATUS "LIBCUDACXX_INCLUDE_DIR: ${LIBCUDACXX_INCLUDE_DIR}")

set(LIBCXX_DIR "${libcudacxx_SOURCE_DIR}/libcxx")
set(LIBCXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/libcxx/include")

message(STATUS "LIBCXX_DIR: ${LIBCXX_DIR}")
message(STATUS "LIBCXX_INCLUDE_DIR: ${LIBCXX_INCLUDE_DIR}")
