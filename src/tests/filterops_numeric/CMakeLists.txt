set(filterops_test_SRCS
    helper/utils.cuh
    helper/utils.cu
 
    test_example.cu
    test_filterops.cu
    test_gpu_concat.cu
)

configure_test(filterops_test "${filterops_test_SRCS}")
