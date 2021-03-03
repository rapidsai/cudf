function(find_and_configure_rdkafka VERSION)
    CPMFindPackage(NAME rdkafka
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/edenhill/librdkafka.git
        GIT_SHALLOW     TRUE
        # SOURCE_SUBDIR   src-cpp
        # OPTIONS         "BUILD_TESTS OFF"
        #                 "BUILD_BENCHMARKS OFF"
        )

    # if(NOT cudf_BINARY_DIR IN_LIST CMAKE_PREFIX_PATH)
    #     list(APPEND CMAKE_PREFIX_PATH "${cudf_BINARY_DIR}")
    #     set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
    # endif()
endfunction()

set(RDKAFKA_VERSION 1.6.1)

find_and_configure_rdkafka(${RDKAFKA_VERSION})