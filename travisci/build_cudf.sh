set -e

if [ "$BUILD_CUDF" == "1" ]; then
  echo "Building cudf"
  CUDF_BUILD_NO_GPU_TEST=1 conda build conda-recipes/cudf -c defaults -c conda-forge -c rapidsai -c numba --python=$PYTHON
fi