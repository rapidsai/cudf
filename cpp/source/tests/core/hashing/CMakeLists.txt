set(hash_test_SRCS
    hash_test.cu
    hash_partition_test.cu
)

configure_test(hash_test "${hash_test_SRCS}")
