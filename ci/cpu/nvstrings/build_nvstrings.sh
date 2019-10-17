set -e

echo "Building nvstrings"
conda build conda/recipes/nvstrings --python=$PYTHON
