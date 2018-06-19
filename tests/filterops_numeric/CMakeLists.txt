include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

CUDA_ADD_EXECUTABLE(Unit_Tests_run
    test_example.cu
    test_with_masks.cu
    test_filterops.cu
    test_filterops_augmented.cu
    utils.cu
)

target_link_libraries(Unit_Tests_run gtest gtest_main)
target_link_libraries(Unit_Tests_run gdf arrow)