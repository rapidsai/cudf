include(FetchContent)

###################################################################################################
# - thrust/cub ------------------------------------------------------------------------------------

set(SRC_THRUST "https://github.com/thrust/thrust.git")

message(STATUS "Fetching thrust from ${SRC_THRUST}")

FetchContent_Declare(
    thrust
    GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
    GIT_TAG        1.10.0
    GIT_SHALLOW    true
    # NOTE: If you change the GIT_TAG you will likely need to change this patch file too
    PATCH_COMMAND  patch -p1 -N < ${CMAKE_CURRENT_SOURCE_DIR}/thrust.patch || true
)

FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
  FetchContent_Populate(thrust)
  # We are not using the thrust CMake targets, so no need to call `add_subdirectory()`.
endif()
set(THRUST_INCLUDE_DIR "${thrust_SOURCE_DIR}" PARENT_SCOPE)

###################################################################################################
# - jitify ----------------------------------------------------------------------------------------

set(SRC_JITIFY "https://github.com/rapidsai/jitify.git")

message(STATUS "Fetching jitify from ${SRC_JITIFY}")

FetchContent_Declare(
    jitify
    GIT_REPOSITORY ${SRC_JITIFY}
    GIT_TAG        cudf_0.16
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

set(SRC_LIBCUDACXX "https://github.com/NVIDIA/libcudacxx.git")

message(STATUS "Fetching libcudacxx from ${SRC_LIBCUDACXX}")

FetchContent_Declare(
    libcudacxx
    GIT_REPOSITORY ${SRC_LIBCUDACXX}
    GIT_TAG        1.4.0
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
