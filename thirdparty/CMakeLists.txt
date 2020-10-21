include(FetchContent)

###################################################################################################
# - jitify ----------------------------------------------------------------------------------------

set(SRC_JITIFY "https://github.com/rapidsai/jitify.git")

message(STATUS "Fetching jitify from ${SRC_JITIFY}")

FetchContent_Declare(
    jitify
    GIT_REPOSITORY ${SRC_JITIFY}
    GIT_TAG        cudf_0.15
    GIT_SHALLOW    true
)

FetchContent_GetProperties(jitify)
if(NOT jitify_POPULATED)
  FetchContent_Populate(jitify)
  # We are not using the jitify CMake targets, so no need to call `add_subdirectory()`.
endif()
set(JITIFY_INCLUDE_DIR "${jitify_SOURCE_DIR}" PARENT_SCOPE)

###################################################################################################
# - libcudacxx ------------------------------------------------------------------------------------

set(SRC_LIBCUDACXX "https://github.com/rapidsai/thirdparty-freestanding.git")

message(STATUS "Fetching libcudacxx from ${SRC_LIBCUDACXX}")

FetchContent_Declare(
    libcudacxx
    GIT_REPOSITORY ${SRC_LIBCUDACXX}
    GIT_TAG        cudf
    GIT_SHALLOW    true
)

FetchContent_GetProperties(libcudacxx)
if(NOT libcudacxx_POPULATED)
  FetchContent_Populate(libcudacxx)
  # libcudacxx has no CMake targets, so no need to call `add_subdirectory()`.
endif()
set(LIBCUDACXX_DIR "${libcudacxx_SOURCE_DIR}" PARENT_SCOPE)
set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include" PARENT_SCOPE)
set(LIBCXX_DIR "${libcudacxx_SOURCE_DIR}/libcxx" PARENT_SCOPE)
set(LIBCXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/libcxx/include" PARENT_SCOPE)
