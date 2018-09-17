set(groupby_test_SRCS
    groupby-kernels-test.cu
    groupby-gdf-test.cu
)

configure_test(groupby_test "${groupby_test_SRCS}")
