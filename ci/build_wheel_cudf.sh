#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cudf"

export SKBUILD_CMAKE_ARGS="-DUSE_LIBARROW_FROM_PYARROW=ON"

# Download the pylibcudf built in the previous step
RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cudf-dep
python -m pip install ./local-cudf-dep/cudf*.whl

./ci/build_wheel.sh ${package_dir}

python -m auditwheel repair -w ${package_dir}/final_dist ${package_dir}/dist/*


RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_dir}/final_dist
