# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from packaging import version

PANDAS_CURRENT_SUPPORTED_VERSION = version.parse("2.3.3")
PANDAS_VERSION = version.parse(pd.__version__)
