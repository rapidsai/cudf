set -e

echo "Building cudf"
conda build conda/recipes/cudf --python=$PYTHON
