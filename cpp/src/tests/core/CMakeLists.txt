###################################################################################################
# - column tests ----------------------------------------------------------------------------------

set(COLUMN_TEST_SRC
    "column/column_test.cu")

ConfigureTest(COLUMN_TEST "${COLUMN_TEST_SRC}")

###################################################################################################
# - cudf tests ------------------------------------------------------------------------------------

set(CUDF_INTERNAL_TEST_SRC
    "cudf/cudf_internal_test.cu")

ConfigureTest(CUDF_INTERNAL_TEST "${CUDF_INTERNAL_TEST_SRC}")

###################################################################################################
# - filter tests ----------------------------------------------------------------------------------

set(FILTER_TEST_SRC
    "filter/helper/utils.cuh"
    "filter/helper/utils.cu"
    "filter/test_example.cu"
    "filter/test_filter_ops.cu"
    "filter/test_gpu_concat.cu")

ConfigureTest(FILTER_TEST "${FILTER_TEST_SRC}")

###################################################################################################
# - groupby tests ---------------------------------------------------------------------------------

set(GROUPBY_TEST_SRC 
    "groupby/groupby_test.cu")

ConfigureTest(GROUPBY_TEST "${GROUPBY_TEST_SRC}")

###################################################################################################
# - join tests ------------------------------------------------------------------------------------

set(JOIN_TEST_SRC 
    "join/join_tests.cu")

ConfigureTest(JOIN_TEST "${JOIN_TEST_SRC}")

###################################################################################################
# - sqls tests ------------------------------------------------------------------------------------

set(SQLS_TEST_SRC 
    "sqls/sqls_test.cu")

ConfigureTest(SQLS_TEST "${SQLS_TEST_SRC}")



