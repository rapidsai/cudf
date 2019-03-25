set -e

if [ "$BUILD_CUDF" == "1" ]; then
  echo "Building cudf"
  export CUDF_BUILD_NO_GPU_TEST=1

  if [ "$BUILD_ABI" == "1" ]; then
    conda build conda/recipes/cudf -c rapidsai -c rapidsai-nightly -c nvidia -c numba -c conda-forge -c defaults --python=$PYTHON
  else
    conda build conda/recipes/cudf -c rapidsai/label/cf201901 -c rapidsai-nightly/label/cf201901 -c nvidia/label/cf201901 -c numba -c conda-forge/label/cf201901 -c defaults --python=$PYTHON
  fi
fi
