set -e

echo "Building cudf"
PYGDF_BUILD_NO_GPU_TEST=1 conda build conda-recipes/pygdf -c defaults -c conda-forge -c rapidsai/label/dev -c numba --python $PYTHON
