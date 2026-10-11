#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading built Conda packages"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")

# CPU runners have no CUDA virtual package, but must install the CUDA runtime dependencies.
export CONDA_OVERRIDE_CUDA="$RAPIDS_CUDA_VERSION"
packages=("${PYTHON_CHANNEL}"/linux-64/*.conda)
if [[ ! -f ${packages[0]} ]]; then
    printf 'No built Python packages found in %s\n' "$PYTHON_CHANNEL" >&2
    exit 1
fi

rapids-logger "Install built packages with only their runtime dependencies"
rapids-mamba-retry create --yes --name python_imports --strict-channel-priority \
    --override-channels \
    --channel "$PYTHON_CHANNEL" --channel "$CPP_CHANNEL" \
    --channel rapidsai-nightly --channel conda-forge \
    "python=${RAPIDS_PY_VERSION}" "${packages[@]}"

set +u
conda activate python_imports
set -u
rapids-print-env

# An installed driver could hide unintended linkage even when no GPU is visible.
python - <<'PY'
import ctypes

for library in ("libcuda.so", "libcuda.so.1"):
    try:
        ctypes.CDLL(library)
    except OSError:
        continue
    raise RuntimeError(f"CPU-only import checks require an unavailable CUDA driver: {library}")
PY

for package in pylibcudf cudf cudf_kafka cudf_streaming; do
    rapids-logger "Import ${package} without a CUDA driver"
    python -c "import ${package}"
done

python -c "import cudf; print(cudf.__version__)"
python -m cudf.pandas -c "import pandas as pd; print(pd)"
python -m cudf.pandas -c "import pandas as pd; print(pd.Series([1, 2, 3]))"
RAPIDS_NO_INITIALIZE=1 python -m cudf.pandas -c "import pandas as pd; print(pd.Series([1, 2, 3]))"
CUDF_NO_INITIALIZE=1 python -m cudf.pandas -c "import pandas as pd; print(pd.Series([1, 2, 3]))"
