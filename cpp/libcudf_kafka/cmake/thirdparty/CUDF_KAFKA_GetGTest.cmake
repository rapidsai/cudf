
CPMFindPackage(NAME GTest
VERSION         1.10.0
GIT_REPOSITORY  https://github.com/google/googletest.git
GIT_TAG         release-1.10.0
GIT_SHALLOW     TRUE
OPTIONS         "INSTALL_GTEST OFF"
# googletest >= 1.10.0 provides a cmake config file -- use it if it exists
FIND_PACKAGE_ARGUMENTS "CONFIG")
# Add GTest aliases if they don't already exist.
# Assumes if GTest::gtest doesn't exist, the others don't either.
# TODO: Is this always a valid assumption?
if(NOT TARGET GTest::gtest)
add_library(GTest::gtest ALIAS gtest)
add_library(GTest::gmock ALIAS gmock)
add_library(GTest::gtest_main ALIAS gtest_main)
add_library(GTest::gmock_main ALIAS gmock_main)
endif()

# include CTest module -- automatically calls enable_testing()
include(CTest)
add_subdirectory(tests)