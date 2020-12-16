# FindThrust
# ---------
#
# Try to find Thrust
#
# Uses Thrust_ROOT in the cache variables or in the environment as a hint where to search
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines a basic `thrust_create_target` function as provided otherwise by the newer
# Thrust >= 1.9.10 included configs.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
# ~~~
# ``Thrust_FOUND`` system has Thrust
# ``Thrust_INCLUDE_DIRS`` the Thrust include directories
# ~~~

include(FindPackageHandleStandardArgs)

# try to find Thrust via installed config first
find_package(Thrust QUIET CONFIG)
if(Thrust_FOUND)
  find_package_handle_standard_args(Thrust CONFIG_MODE)
  return()
endif()

cmake_minimum_required(VERSION 3.17..3.18 FATAL_ERROR)

find_dependency(CUDAToolkit)

find_path(
  Thrust_INCLUDE_DIRS
  NAMES thrust/version.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS})

file(READ ${Thrust_INCLUDE_DIRS}/thrust/version.h _version_header)
string(REGEX MATCH "#define THRUST_VERSION ([0-9]*)" _match "${_version_header}")
math(EXPR major "${CMAKE_MATCH_1} / 100000")
math(EXPR minor "(${CMAKE_MATCH_1} / 100) % 1000")
math(EXPR subminor "${CMAKE_MATCH_1} % 100")
set(Thrust_VERSION "${major}.${minor}.${subminor}")

find_package_handle_standard_args(
  Thrust
  REQUIRED_VARS Thrust_INCLUDE_DIRS
  VERSION_VAR Thrust_VERSION)

if(Thrust_FOUND)
  # Create wrapper function to handle situation where we can't use a regular IMPORTED INTERFACE
  # target since that'll use -isystem, leading to the wrong search order with nvcc
  function(thrust_create_target tgt)
    if(NOT TARGET ${tgt})
      add_library(thrust_internal INTERFACE)
      set_target_properties(thrust_internal PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                       "${Thrust_INCLUDE_DIRS}")
      add_library(${tgt} ALIAS thrust_internal)
    endif()
  endfunction()
endif()
