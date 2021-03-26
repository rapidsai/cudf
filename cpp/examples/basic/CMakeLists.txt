cmake_minimum_required(VERSION 3.18)

project(libcudf_example VERSION 0.0.1 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CPM_DOWNLOAD_VERSION 0.27.2) 
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(NAME  cudf
    GIT_REPOSITORY  https://github.com/rapidsai/cudf
    GIT_TAG         branch-0.19
    GIT_SHALLOW     TRUE
    SOURCE_SUBDIR   cpp
    OPTIONS         "BUILD_TESTS OFF"
                    "BUILD_BENCHMARKS OFF"
                    "ARROW_STATIC_LIB ON"
                    "JITIFY_USE_CACHE ON"
                    "CUDA_STATIC_RUNTIME ON"
                    "DISABLE_DEPRECATION_WARNING ON"
)

# Configure your project here
add_executable(${PROJECT_NAME} "src/process_csv.cpp")
target_link_libraries(${PROJECT_NAME} cudf::cudf)
