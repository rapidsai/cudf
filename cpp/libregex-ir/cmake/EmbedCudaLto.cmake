# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Copyright (c) 2026, Regex IR contributors. SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Compile a CUDA source fragment to LTO-IR and embed its fatbin in a generated include.
function(regex_ir_embed_cuda_lto output_variable variant source symbol)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "regex_ir_embed_cuda_lto requires the CUDA language")
  endif()

  if(NOT REGEX_IR_BIN2C_EXECUTABLE)
    find_program(
      REGEX_IR_BIN2C_EXECUTABLE
      NAMES bin2c
      HINTS ${CUDAToolkit_BIN_DIR} REQUIRED
    )
  endif()

  set(output_directory ${CMAKE_CURRENT_BINARY_DIR}/generated)
  set(fatbin ${output_directory}/${variant}.fatbin)
  set(embedded ${output_directory}/${variant}.fatbin.inc)
  add_custom_command(
    OUTPUT ${fatbin}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${output_directory}
    COMMAND ${CMAKE_CUDA_COMPILER} --fatbin --dlink-time-opt --gen-opt-lto
            --relocatable-device-code=true --std=c++20 -o ${fatbin} ${source}
    DEPENDS ${source}
    COMMENT "Compiling ${variant} to CUDA LTO-IR fatbin"
    VERBATIM
  )
  add_custom_command(
    OUTPUT ${embedded}
    COMMAND ${CMAKE_COMMAND} -DBIN2C=${REGEX_IR_BIN2C_EXECUTABLE} -DINPUT=${fatbin}
            -DOUTPUT=${embedded} -DSYMBOL=${symbol} -P ${PROJECT_SOURCE_DIR}/cmake/bin2c.cmake
    DEPENDS ${fatbin} ${PROJECT_SOURCE_DIR}/cmake/bin2c.cmake
    COMMENT "Embedding ${variant} LTO-IR fatbin"
    VERBATIM
  )

  set(${output_variable}
      ${embedded}
      PARENT_SCOPE
  )
endfunction()
