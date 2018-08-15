set -e

if [ "$BUILD_CFFI" == "1" ]; then
    echo "Building libgdf_cffi"
    conda build conda-recipes/libgdf_cffi -c defaults -c conda-forge -c numba --python=${PYTHON}
fi
