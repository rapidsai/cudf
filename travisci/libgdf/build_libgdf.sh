set -e

if [ $BUILD_LIBGDF == '1' ]; then
    echo "Building libgdf"
    conda build conda-recipes/libgdf -c defaults -c conda-forge -c numba
fi
