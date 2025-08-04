# CMake functions for generating LTO-IR from CUDA kernels
#
# This file contains CMake functions that would be used to pre-compile
# CUDA kernels to LTO-IR during the build process.

# Function to compile a CUDA kernel to LTO-IR
function(generate_lto_ir_operator)
  cmake_parse_arguments(
    ARG
    ""
    "TARGET;KERNEL_NAME;SOURCE_FILE;OUTPUT_VAR"
    "INCLUDE_DIRS;COMPILE_FLAGS"
    ${ARGN}
  )

  if(NOT CUDF_USE_LTO_IR)
    return()
  endif()

  # Generate LTO-IR output filename
  get_filename_component(SOURCE_NAME ${ARG_SOURCE_FILE} NAME_WE)
  set(LTO_IR_FILE "${CMAKE_CURRENT_BINARY_DIR}/lto_ir/${SOURCE_NAME}_${ARG_KERNEL_NAME}.lto.o")

  # Create the LTO-IR generation command
  add_custom_command(
    OUTPUT ${LTO_IR_FILE}
    COMMAND ${CMAKE_CUDA_COMPILER}
      --device-lto
      --device-c
      ${ARG_COMPILE_FLAGS}
      -I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
      $<$<BOOL:${ARG_INCLUDE_DIRS}>:-I$<JOIN:${ARG_INCLUDE_DIRS}, -I>>
      -o ${LTO_IR_FILE}
      ${ARG_SOURCE_FILE}
    DEPENDS ${ARG_SOURCE_FILE}
    COMMENT "Generating LTO-IR for ${ARG_KERNEL_NAME}"
    VERBATIM
  )

  # Create a target for this LTO-IR file if one doesn't exist
  set(LTO_IR_TARGET "${ARG_TARGET}_lto_ir")
  if(NOT TARGET ${LTO_IR_TARGET})
    add_custom_target(${LTO_IR_TARGET})
  endif()

  add_custom_target(${ARG_KERNEL_NAME}_lto_ir DEPENDS ${LTO_IR_FILE})
  add_dependencies(${LTO_IR_TARGET} ${ARG_KERNEL_NAME}_lto_ir)

  # Set the output variable
  if(ARG_OUTPUT_VAR)
    set(${ARG_OUTPUT_VAR} ${LTO_IR_FILE} PARENT_SCOPE)
  endif()
endfunction()

# Function to generate LTO-IR data source file
function(generate_lto_ir_data_file)
  cmake_parse_arguments(
    ARG
    ""
    "TARGET;OUTPUT_FILE"
    "LTO_IR_FILES;OPERATOR_NAMES"
    ${ARGN}
  )

  if(NOT CUDF_USE_LTO_IR)
    return()
  endif()

  # Generate a C++ source file containing the LTO-IR data
  set(GENERATED_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/lto_ir_data.cpp")
  
  add_custom_command(
    OUTPUT ${GENERATED_SOURCE}
    COMMAND ${CMAKE_COMMAND}
      -D "LTO_IR_FILES=${ARG_LTO_IR_FILES}"
      -D "OPERATOR_NAMES=${ARG_OPERATOR_NAMES}"
      -D "OUTPUT_FILE=${GENERATED_SOURCE}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_lto_ir_data.cmake"
    DEPENDS ${ARG_LTO_IR_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_lto_ir_data.cmake"
    COMMENT "Generating LTO-IR data source file"
    VERBATIM
  )

  target_sources(${ARG_TARGET} PRIVATE ${GENERATED_SOURCE})
endfunction()

# Function to automatically generate LTO-IR for common operators
function(generate_builtin_lto_ir_operators TARGET)
  if(NOT CUDF_USE_LTO_IR)
    return()
  endif()

  set(KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/binaryop/jit/kernel.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/jit/kernel.cu"
  )

  set(OPERATOR_NAMES
    "binary_op::add"
    "binary_op::subtract" 
    "binary_op::multiply"
    "binary_op::divide"
    "binary_op::equal"
    "binary_op::not_equal"
    "binary_op::less"
    "binary_op::greater"
    "binary_op::less_equal"
    "binary_op::greater_equal"
    "binary_op::logical_and"
    "binary_op::logical_or"
    "transform::sin"
    "transform::cos"
    "transform::exp"
    "transform::log"
    "transform::sqrt"
    "transform::abs"
  )

  set(LTO_IR_FILES)
  
  # Generate LTO-IR for each operator
  foreach(SOURCE IN LISTS KERNEL_SOURCES)
    get_filename_component(SOURCE_NAME ${SOURCE} NAME_WE)
    
    generate_lto_ir_operator(
      TARGET ${TARGET}
      KERNEL_NAME ${SOURCE_NAME}
      SOURCE_FILE ${SOURCE}
      OUTPUT_VAR LTO_IR_FILE
      INCLUDE_DIRS 
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
      COMPILE_FLAGS
        "-arch=sm_70"  # Target common architecture
        "--device-int128"
        "-DCUDF_USE_LTO_IR"
    )
    
    list(APPEND LTO_IR_FILES ${LTO_IR_FILE})
  endforeach()

  # Generate the data file containing all LTO-IR
  generate_lto_ir_data_file(
    TARGET ${TARGET}
    LTO_IR_FILES "${LTO_IR_FILES}"
    OPERATOR_NAMES "${OPERATOR_NAMES}"
  )
endfunction()