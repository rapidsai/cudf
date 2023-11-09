# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

set(cython_sources column.pyx copying.pyx gpumemoryview.pyx interop.pyx scalar.pyx table.pyx
                   types.pyx utils.pyx
)
set(linked_libraries cudf::cudf)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX pylibcudf_ ASSOCIATED_TARGETS cudf
)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pyarrow; print(pyarrow.get_include())"
  OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

foreach(target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  target_include_directories(${target} PRIVATE "${PYARROW_INCLUDE_DIR}")
endforeach()

# TODO: Clean up this include when switching to scikit-build-core. See cudf/_lib/CMakeLists.txt for
# more info
find_package(NumPy REQUIRED)
foreach(target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  target_include_directories(${target} PRIVATE "${NumPy_INCLUDE_DIRS}")
  # Switch to the line below when we switch back to FindPython.cmake in CMake 3.24.
  # target_include_directories(${target} PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
endforeach()
