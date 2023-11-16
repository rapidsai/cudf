# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

set(cython_sources kafka.pyx)
set(linked_libraries cudf_kafka::cudf_kafka)

rapids_cython_create_modules(
  CXX ASSOCIATED_TARGETS cudf_kafka
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
)

# TODO: Finding NumPy currently requires finding Development due to a bug in CMake. This bug was
# fixed in https://gitlab.kitware.com/cmake/cmake/-/merge_requests/7410 and will be available in
# CMake 3.24, so we can remove the Development component once we upgrade to CMake 3.24.
# find_package(Python REQUIRED COMPONENTS Development NumPy)

# Note: The bug noted above prevents us from finding NumPy successfully using FindPython.cmake
# inside the manylinux images used to build wheels because manylinux images do not contain
# libpython.so and therefore Development cannot be found. Until we upgrade to CMake 3.24, we should
# use FindNumpy.cmake instead (provided by scikit-build). When we switch to 3.24 we can try
# switching back, but it may not work if that implicitly still requires Python libraries. In that
# case we'll need to follow up with the CMake team to remove that dependency.  The stopgap solution
# is to unpack the static lib tarballs in the wheel building jobs so that there are at least static
# libs to be found, but that should be a last resort since it implies a dependency that isn't really
# necessary. The relevant command is tar -xf /opt/_internal/static-libs-for-embedding-only.tar.xz -C
# /opt/_internal"
find_package(NumPy REQUIRED)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pyarrow; print(pyarrow.get_include())"
  OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
  ERROR_VARIABLE PYARROW_ERROR
  RESULT_VARIABLE PYARROW_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(${PYARROW_RESULT})
  message(FATAL_ERROR "Error while trying to obtain pyarrow include directory:\n${PYARROW_ERROR}")
endif()

# TODO: Due to cudf's scalar.pyx needing to cimport pylibcudf's scalar.pyx (because there are parts
# of cudf Cython that need to directly access the c_obj underlying the pylibcudf Scalar) the
# requirement for arrow headers infects all of cudf. That in turn requires including numpy headers.
# These requirements will go away once all scalar-related Cython code is removed from cudf.
foreach(target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  target_include_directories(${target} PRIVATE "${NumPy_INCLUDE_DIRS}")
  target_include_directories(${target} PRIVATE "${PYARROW_INCLUDE_DIR}")
endforeach()
