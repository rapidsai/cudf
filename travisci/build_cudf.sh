set -e

echo "Building cudf"
CUDF_BUILD_NO_GPU_TEST=1 conda build conda-recipes/cudf -c defaults -c conda-forge -c rapidsai/label/dev -c numba --python=$PYTHON