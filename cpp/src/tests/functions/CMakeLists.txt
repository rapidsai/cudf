###################################################################################################
# - bitmask tests ---------------------------------------------------------------------------------

set(BITMASK_TEST_SRC
    "bitmask/valid_ops_test.cu")

ConfigureTest(BITMASK_TEST "${BITMASK_TEST_SRC}")

###################################################################################################
# - datetime tests --------------------------------------------------------------------------------

set(DATETIME_TEST_SRC
    "datetime/datetime_ops_test.cu")

ConfigureTest(DATETIME_TEST "${DATETIME_TEST_SRC}")

###################################################################################################
# - hashing tests ---------------------------------------------------------------------------------

set(HASHING_TEST_SRC
    "hashing/hash_partition_test.cu"
    "hashing/hash_test.cu")

ConfigureTest(HASHING_TEST "${HASHING_TEST_SRC}")

###################################################################################################
# - hash_map tests --------------------------------------------------------------------------------

set(HASH_MAP_TEST_SRC 
    "hash_map/map_test.cu"
    "hash_map/multimap_test.cu")

ConfigureTest(HASH_MAP_TEST "${HASH_MAP_TEST_SRC}")

###################################################################################################
# - quantiles tests -------------------------------------------------------------------------------

set(QUANTILES_TEST_SRC 
    "quantiles/quantiles_test.cu")

ConfigureTest(QUANTILES_TEST "${QUANTILES_TEST_SRC}")

###################################################################################################
# - unary tests -----------------------------------------------------------------------------------

set(UNARY_TEST_SRC 
    "unary/unary_ops_test.cu")

ConfigureTest(UNARY_TEST "${UNARY_TEST_SRC}")



