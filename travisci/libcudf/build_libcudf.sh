set -e

if [ "$BUILD_LIBCUDF" == '1' -o "$BUILD_CFFI" == '1' ]; then
  echo "Building libcudf"
  conda build conda/recipes/libcudf -c nvidia -c rapidsai -c numba -c conda-forge -c defaults
fi
