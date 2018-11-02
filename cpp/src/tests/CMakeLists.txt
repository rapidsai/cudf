cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDF_TESTS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQURIED ON)
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# suppress SHFL warnings caused by Modern GPU
# TODO: remove this when Modern GPU is removed or fixed to use shfl_sync
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Wno-deprecated-declarations -Xptxas --disable-warnings")

# set warnings as errors
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Werror")

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
	add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SRC})
	target_link_libraries(${CMAKE_TEST_NAME} gmock_main gmock GTest::GTest cudf cudart arrow)
	set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
							RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/gtests")
	add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/.."
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
					"${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/cub"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/moderngpu/src"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/cnmem/include")

###################################################################################################
# - source paths ----------------------------------------------------------------------------------

add_subdirectory(core)
#add_subdirectory(functions)
#add_subdirectory(io)
#add_subdirectory(rmm)
#add_subdirectory(utilities)

###################################################################################################
# - enable testing --------------------------------------------------------------------------------

enable_testing()







