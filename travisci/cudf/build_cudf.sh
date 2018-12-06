set -e

if [ "$BUILD_CUDF" == "1" ]; then
  echo "Building cudf"
  export CUDF_BUILD_NO_GPU_TEST=1 
  conda build conda/recipes/cudf -c nvidia -c rapidsai -c numba -c conda-forge -c defaults --python=$PYTHON
fi
