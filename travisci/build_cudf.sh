set -e

if [ "$BUILD_CUDF" == "1" ]; then
  echo "Building cudf"
  export CUDF_BUILD_NO_GPU_TEST=1 
  travis_retry conda build conda-recipes/cudf -c defaults -c conda-forge -c nvidia -c rapidsai -c numba --python=$PYTHON
fi