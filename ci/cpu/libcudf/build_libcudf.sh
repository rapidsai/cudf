set -e

if [ "$BUILD_LIBCUDF" == '1' -o "$BUILD_CFFI" == '1' ]; then
  echo "Building libcudf"
  CUDA_REL=${CUDA:0:3}
  if [ "${CUDA:0:2}" == '10' ]; then
    # CUDA 10 release
    CUDA_REL=${CUDA:0:4}
  fi

  if [ "$BUILD_ABI" == "1" ]; then
    conda build conda/recipes/libcudf -c nvidia/label/cuda${CUDA_REL} -c rapidsai/label/cuda${CUDA_REL} -c numba -c conda-forge -c defaults
  else
    conda build conda/recipes/libcudf -c nvidia/label/cf201901-cuda${CUDA_REL} -c rapidsai/label/cf201901-cuda${CUDA_REL} -c numba -c conda-forge/label/cf201901 -c defaults
  fi
fi
