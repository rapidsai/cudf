set(filterops_test_SRCS
    helper/utils.cuh
    helper/utils.cu

    #test_example.cu
    #test_with_masks.cu
    test_gpu_concat.cu
    #test_filterops.cu
    #test_filterops_augmented.cu

)

configure_test(filterops_test "${filterops_test_SRCS}")
