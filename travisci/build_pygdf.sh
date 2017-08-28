set -e

echo "Building pygdf"
conda build conda-recipes/pygdf -c defaults -c conda-forge -c gpuopenanalytics/label/dev -c numba --python $PYTHON
