include(FetchContent)

###################################################################################################
# - thrust/cub ------------------------------------------------------------------------------------

message("Fetching Thrust")

CPMFindPackage(
  NAME Thrust
  GITHUB_REPOSITORY NVIDIA/thrust
  GIT_TAG 1.10.0
  VERSION 1.10.0
  GIT_SHALLOW TRUE
  OPTIONS
    # If there is no pre-installed thrust we can use, we'll install our fetched copy
    # together with CUDF
    "THRUST_INSTALL TRUE"
  )

thrust_create_target(cudf::Thrust FROM_OPTIONS)

###################################################################################################
# - jitify ----------------------------------------------------------------------------------------

message("Fetching Jitify")

FetchContent_Declare(
    jitify
    GIT_REPOSITORY https://github.com/rapidsai/jitify.git
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

message("Fetching libcudacxx")

FetchContent_Declare(
    libcudacxx
    GIT_REPOSITORY https://github.com/rapidsai/thirdparty-freestanding.git
    GIT_TAG        cudf
    GIT_SHALLOW    true
)

FetchContent_GetProperties(libcudacxx)
if(NOT libcudacxx_POPULATED)
  FetchContent_Populate(libcudacxx)
  # libcudacxx has no CMake targets, so no need to call `add_subdirectory()`.
endif()
set(LIBCUDACXX_DIR "${libcudacxx_SOURCE_DIR}" PARENT_SCOPE)
set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include/" PARENT_SCOPE)
set(LIBCXX_DIR "${libcudacxx_SOURCE_DIR}/libcxx/" PARENT_SCOPE)
set(LIBCXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/libcxx/include/" PARENT_SCOPE)
