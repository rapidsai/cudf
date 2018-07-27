set(join_test_SRCS
    multi-left-join-test.cu
    join-tests.cu
)

configure_test(join_test "${join_test_SRCS}")
