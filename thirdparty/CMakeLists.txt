include(FetchContent)

###################################################################################################
# - thrust/cub ------------------------------------------------------------------------------------

CPMFindPackage(
  NAME Thrust
  GITHUB_REPOSITORY NVIDIA/thrust
  GIT_TAG 1.10.0
  VERSION 1.10.0
  GIT_SHALLOW TRUE
)

thrust_create_target(cudf::Thrust FROM_OPTIONS)

###################################################################################################
# - jitify ----------------------------------------------------------------------------------------

CPMFindPackage(
  NAME jitify
  GITHUB_REPOSITORY rapidsai/jitify
  GIT_TAG cudf_0.15
  VERSION 0.15.0
  GIT_SHALLOW TRUE
)

set(JITIFY_INCLUDE_DIR "${jitify_SOURCE_DIR}" PARENT_SCOPE)

###################################################################################################
# - libcudacxx ------------------------------------------------------------------------------------

CPMFindPackage(
  NAME libcudacxx
  GITHUB_REPOSITORY rapidsai/thirdparty-freestanding
  GIT_TAG cudf
  GIT_SHALLOW TRUE
)

set(LIBCUDACXX_DIR "${libcudacxx_SOURCE_DIR}" PARENT_SCOPE)
set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include" PARENT_SCOPE)
set(LIBCXX_DIR "${libcudacxx_SOURCE_DIR}/libcxx" PARENT_SCOPE)
set(LIBCXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/libcxx/include" PARENT_SCOPE)
