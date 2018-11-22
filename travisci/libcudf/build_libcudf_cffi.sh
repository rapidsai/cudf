set -e

# Patch tests for CUDA 10 skipping hash_map
## See https://github.com/rapidsai/libgdf/pull/149
if [ ${CUDA:0:4} == '10.0' ]; then
  echo "CUDA 10 detected, removing hash_map tests"
  sed -i.bak 's/ConfigureTest(HASH_MAP_TEST "${HASH_MAP_TEST_SRC}")/#ConfigureTest(HASH_MAP_TEST "${HASH_MAP_TEST_SRC}")/g' ./cpp/tests/CMakeLists.txt
fi

if [ "$BUILD_CFFI" == '1' ]; then
  echo "Building cudf_cpp_cffi"
  conda build conda/recipes/libcudf_cffi -c nvidia -c rapidsai -c numba -c conda-forge -c defaults --python=${PYTHON}
fi
