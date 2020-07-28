set -e

echo "Building cudf_kafka"
CUDA_REL=${CUDA_VERSION%.*}

conda build conda/recipes/cudf_kafka
