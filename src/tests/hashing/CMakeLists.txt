set(hash_test_SRCS
    hash-test.cu
    hash-partition-test.cu
)

configure_test(hash_test "${hash_test_SRCS}")
