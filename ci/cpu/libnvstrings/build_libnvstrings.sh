set -e

echo "Building libNVStrings"
CUDA_REL=${CUDA_VERSION%.*}

conda build conda/recipes/libnvstrings
