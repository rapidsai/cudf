set -e

echo "Building cudf"
export CUDF_BUILD_NO_GPU_TEST=1

conda build conda/recipes/cudf --python=$PYTHON
