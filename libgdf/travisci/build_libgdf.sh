set -e

echo "Building libgdf"
conda build conda-recipes/libgdf -c defaults -c conda-forge -c numba
