set -e

echo "Building libcudf"
CUDA_REL=${CUDA_VERSION%.*}

conda build conda/recipes/libcudf
