set -e

echo "Building libcudf_kafka"
CUDA_REL=${CUDA_VERSION%.*}

conda build conda/recipes/libcudf_kafka
