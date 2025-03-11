#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test_python_cudf.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1;

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh test_python_cudf

rapids-print-env
# shellcheck disable=SC2034
EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "import cudf"
python -c "import cudf"
rapids-logger "import cudf.pandas"
python -m cudf.pandas -c "import pandas as pd;print(pd)"
rapids-logger "import cudf.pandas and construct a Series"
python -m cudf.pandas -c "import pandas as pd;print(pd.Series([1, 2, 3]))"
rapids-logger "import cudf.pandas with RAPIDS_NO_INITIALIZE and construct a Series"
RAPIDS_NO_INITIALIZE=1 python -m cudf.pandas -c "import pandas as pd;print(pd.Series([1, 2, 3]))"
