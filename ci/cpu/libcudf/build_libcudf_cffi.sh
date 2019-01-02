set -e

if [ "$BUILD_CFFI" == '1' ]; then
  echo "Building libcudf_cffi"
  conda build conda/recipes/libcudf_cffi -c nvidia -c rapidsai -c numba -c conda-forge -c defaults --python=${PYTHON}
fi
