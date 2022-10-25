# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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
include_guard(GLOBAL)

# Compile protobuf files to Python. All arguments are assumed to be .proto files.
function(codegen_protoc)
  # Allow user to provide path to protoc executable as an environment variable.
  if(DEFINED ENV{PROTOC})
    set(protoc_COMMAND $ENV{PROTOC})
  else()
    find_program(protoc_COMMAND protoc REQUIRED)
  endif()

  foreach(_proto_path IN LISTS ARGV)
    string(REPLACE "\.proto" "_pb2\.py" pb2_py_path "${_proto_path}")
    set(pb2_py_path "${CMAKE_CURRENT_SOURCE_DIR}/${pb2_py_path}")
    # Note: If we ever need to process larger numbers of protobuf files we should consider switching
    # to protobuf_generate_python from the FindProtobuf module.
    execute_process(
      COMMAND ${protoc_COMMAND} --python_out=. "${_proto_path}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMMAND_ERROR_IS_FATAL ANY
    )
    # Mark entire file to skip formatting.
    file(READ "${pb2_py_path}" pb2_py)
    file(
      WRITE "${pb2_py_path}"
      [=[
# flake8: noqa
# fmt: off
]=]
    )
    file(APPEND "${pb2_py_path}" "${pb2_py}")
    file(
      APPEND "${pb2_py_path}"
      [=[
# fmt: on
]=]
    )
  endforeach()
endfunction()
