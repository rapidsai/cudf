set(join_test_SRCS
    multi-left-join-test.cu
    join-tests.cu
)

set(new_join_test_SRCS
    join-tests.cu
)

configure_test(join_test "${join_test_SRCS}")
configure_test(new_join_test "${new_join_test_SRCS}")
