set(filterops_test_SRCS
    test_example.cu
    test_with_masks.cu
    test_filterops.cu
    test_filterops_augmented.cu
    utils.cu
)

configure_test(filterops_test "${filterops_test_SRCS}")
