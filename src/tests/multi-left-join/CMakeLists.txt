set(multi-left-join_test_SRCS
    multi-left-join-test.cu
    join-tests.cu
)

configure_test(multi-left-join_test "${multi-left-join_test_SRCS}")
