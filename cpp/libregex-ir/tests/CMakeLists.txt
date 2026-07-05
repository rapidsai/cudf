find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

find_path(REGEX_IR_NVVM_INCLUDE_DIR
  NAMES nvvm.h
  HINTS ${CUDAToolkit_LIBRARY_ROOT}/nvvm/include
  REQUIRED)
find_library(REGEX_IR_NVVM_LIBRARY
  NAMES nvvm
  HINTS ${CUDAToolkit_LIBRARY_ROOT}/nvvm/lib64
  REQUIRED)
find_path(REGEX_IR_NVJITLINK_INCLUDE_DIR
  NAMES nvJitLink.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
  REQUIRED)
find_library(REGEX_IR_NVJITLINK_LIBRARY
  NAMES nvJitLink
  HINTS ${CUDAToolkit_LIBRARY_DIR}
        ${CUDAToolkit_LIBRARY_ROOT}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib
  REQUIRED)
include(${PROJECT_SOURCE_DIR}/cmake/EmbedCudaLto.cmake)

set(REGEX_IR_KERNEL_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(REGEX_IR_FRAGMENT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/fragments)
regex_ir_embed_cuda_lto(
  REGEX_IR_BOOLEAN_KERNEL
  regex_ir_boolean_kernel
  ${REGEX_IR_FRAGMENT_DIRECTORY}/regex_ir_boolean_kernel.cu
  regex_ir_boolean_kernel_fatbin)
regex_ir_embed_cuda_lto(
  REGEX_IR_CAPTURE_KERNEL
  regex_ir_capture_kernel
  ${REGEX_IR_FRAGMENT_DIRECTORY}/regex_ir_capture_kernel.cu
  regex_ir_capture_kernel_fatbin)
regex_ir_embed_cuda_lto(
  REGEX_IR_FIND_KERNEL
  regex_ir_find_kernel
  ${REGEX_IR_FRAGMENT_DIRECTORY}/regex_ir_find_kernel.cu
  regex_ir_find_kernel_fatbin)
regex_ir_embed_cuda_lto(
  REGEX_IR_COUNT_KERNEL
  regex_ir_count_kernel
  ${REGEX_IR_FRAGMENT_DIRECTORY}/regex_ir_count_kernel.cu
  regex_ir_count_kernel_fatbin)
regex_ir_embed_cuda_lto(
  REGEX_IR_REPLACE_KERNEL
  regex_ir_replace_kernel
  ${REGEX_IR_FRAGMENT_DIRECTORY}/regex_ir_replace_kernel.cu
  regex_ir_replace_kernel_fatbin)
regex_ir_embed_cuda_lto(
  REGEX_IR_SPLIT_KERNEL
  regex_ir_split_kernel
  ${REGEX_IR_FRAGMENT_DIRECTORY}/regex_ir_split_kernel.cu
  regex_ir_split_kernel_fatbin)
set(REGEX_IR_EMBEDDED_KERNELS
  ${REGEX_IR_BOOLEAN_KERNEL}
  ${REGEX_IR_CAPTURE_KERNEL}
  ${REGEX_IR_FIND_KERNEL}
  ${REGEX_IR_COUNT_KERNEL}
  ${REGEX_IR_REPLACE_KERNEL}
  ${REGEX_IR_SPLIT_KERNEL})

add_custom_target(regex_ir_test_kernels DEPENDS ${REGEX_IR_EMBEDDED_KERNELS})

add_executable(regex_ir_tests unit_tests.cpp)
add_dependencies(regex_ir_tests regex_ir_test_kernels)
target_include_directories(regex_ir_tests PRIVATE
  ${REGEX_IR_KERNEL_DIRECTORY}
  ${REGEX_IR_NVVM_INCLUDE_DIR}
  ${REGEX_IR_NVJITLINK_INCLUDE_DIR})
target_link_libraries(regex_ir_tests PRIVATE regex_ir::regex_ir GTest::gtest)
target_link_libraries(regex_ir_tests PRIVATE
  CUDA::cuda_driver
  ${REGEX_IR_NVVM_LIBRARY}
  ${REGEX_IR_NVJITLINK_LIBRARY}
  ${CMAKE_DL_LIBS})
target_compile_features(regex_ir_tests PRIVATE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(regex_ir_tests PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

include(GoogleTest)
gtest_discover_tests(regex_ir_tests TEST_PREFIX "regex_ir.unit.")

if(TARGET regex_ir_explorer)
  foreach(REGEX_IR_EXPLORER_OPERATION IN ITEMS contains match find count extract split)
    add_test(NAME regex_ir.explorer_${REGEX_IR_EXPLORER_OPERATION}
      COMMAND regex_ir_explorer --ir
              --operation ${REGEX_IR_EXPLORER_OPERATION} "a(b|c)+")
  endforeach()
  add_test(NAME regex_ir.explorer_replace
    COMMAND regex_ir_explorer --ir --operation replace
            --replacement "<$1>" "a(b|c)+")
  add_test(NAME regex_ir.explorer_contains_execute
    COMMAND regex_ir_explorer --ir [=[contains("[0-9]", 12834)]=])
  set_tests_properties(regex_ir.explorer_contains_execute
    PROPERTIES PASS_REGULAR_EXPRESSION "contains = true")
  add_test(NAME regex_ir.explorer_help COMMAND regex_ir_explorer --help)
endif()

if(TARGET regex_ir_codegen)
  add_test(NAME regex_ir.codegen_nvvm
    COMMAND regex_ir_codegen --execute-function smoke_execute "a(b|c)+")
  add_test(NAME regex_ir.codegen_count
    COMMAND regex_ir_codegen --operation count "a+")
  add_test(NAME regex_ir.codegen_replace
    COMMAND regex_ir_codegen --replace "<$1>" "(a+)")
  add_test(NAME regex_ir.codegen_split
    COMMAND regex_ir_codegen --operation split "[,;]+")
  add_test(NAME regex_ir.codegen_empty COMMAND regex_ir_codegen "")
  add_test(NAME regex_ir.codegen_help COMMAND regex_ir_codegen --help)
endif()

add_executable(regex_ir_fuzz_smoke fuzz_tests.cpp)
target_compile_definitions(regex_ir_fuzz_smoke PRIVATE REGEX_IR_FUZZ_SMOKE=1)
target_link_libraries(regex_ir_fuzz_smoke PRIVATE regex_ir::regex_ir)
target_compile_features(regex_ir_fuzz_smoke PRIVATE cxx_std_20)
add_test(NAME regex_ir.fuzz_smoke COMMAND regex_ir_fuzz_smoke)
