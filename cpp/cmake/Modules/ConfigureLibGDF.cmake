#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

# BEGIN macros

macro(CONFIGURE_GPU_LIBGDF_EXTERNAL_PROJECT)
    # NOTE Reset arrow env vars since libgdf has its own arrow dependency
    set(ENV{BOOST_ROOT} ${BOOST_ROOT})
    set(ENV{FLATBUFFERS_HOME} "")
    set(ENV{LZ4_HOME} "")
    set(ENV{ZSTD_HOME} "")
    set(ENV{BROTLI_HOME} "")
    set(ENV{SNAPPY_HOME} "")
    set(ENV{THRIFT_HOME} "")
    set(ENV{PARQUET_HOME} "")

    # NOTE Define basic env vars to build libgdf
    set(ENV{BOOST_ROOT} ${BOOST_ROOT})
    set(ENV{CUDACXX} $ENV{CUDACXX})
    set(ENV{NVSTRINGS_ROOT} ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-download/nvstrings-prefix/src/nvstrings/)
    set(NVSTRINGS_INSTALL_DIR $ENV{NVSTRINGS_ROOT})

    # TODO pass ARROW_INSTALL_DIR when cudf support vendored arrow builds

    # Download and unpack libgdf at configure time
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/LibGDF.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-download/CMakeLists.txt)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-download/
    )

    if(result)
        message(FATAL_ERROR "CMake step for libgdf failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -- -j8
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for libgdf failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

if (LIBGDF_INSTALL_DIR)
    if (NOT NVSTRINGS_INSTALL_DIR)
        message(FATAL_ERROR "If you use the LIBGDF_INSTALL_DIR argument then you need pass the NVSTRINGS_INSTALL_DIR argument too (the home installation of nvstrings)")
    endif()

    message(STATUS "LIBGDF_INSTALL_DIR defined, it will use vendor version from ${LIBGDF_INSTALL_DIR}")
    set(LIBGDF_ROOT "${LIBGDF_INSTALL_DIR}")
else()
    message(STATUS "LIBGDF_INSTALL_DIR not defined, it will be built from sources")
    configure_gpu_libgdf_external_project()
    set(LIBGDF_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-install/")

    # TODO percy remove these lines when cudf properly install rmm headers
    configure_file(${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-src/cpp/src/rmm/memory.h ${LIBGDF_ROOT}/include/memory.h COPYONLY)
    configure_file(${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/libgdf-src/cpp/src/rmm/rmm.h ${LIBGDF_ROOT}/include/rmm.h COPYONLY)
endif()

set(NVSTRINGS_LIBDIR ${NVSTRINGS_INSTALL_DIR}/lib/)
link_directories(${NVSTRINGS_LIBDIR})

find_package(LibGDF REQUIRED)
set_package_properties(LibGDF PROPERTIES TYPE REQUIRED
    PURPOSE "libgdf is a C library for implementing common functionality for a GPU Data Frame."
    URL "https://github.com/gpuopenanalytics/libgdf")

if(NOT LIBGDF_FOUND)
    message(FATAL_ERROR "libgdf not found, please check your settings.")
endif()

message(STATUS "libgdf found in ${LIBGDF_ROOT}")

include_directories(${LIBGDF_INCLUDEDIR} ${LIBGDF_INCLUDE_DIR})
# TODO percy seems cmake bug: we cannot define target dirs per cuda target
# ... see if works in future cmake versions
link_directories(${LIBGDF_LIBDIR})

# END MAIN #
