# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def filter_pandas_versions(
    available_pandas_versions: list[str], pandas_range: str
) -> list[str]:
    matching_versions = map(
        Version,
        SpecifierSet(pandas_range.lstrip("pandas")).filter(
            available_pandas_versions
        ),
    )
    return sorted(
        set(".".join((str(v.major), str(v.minor))) for v in matching_versions),
        key=Version,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter pandas versions by prefix."
    )
    parser.add_argument(
        "available_pandas_versions",
        type=json.loads,
        help="The available pandas versions.",
    )
    parser.add_argument(
        "pandas_range", type=str, help="The version prefix to filter by."
    )
    args = parser.parse_args()

    versions = filter_pandas_versions(
        args.available_pandas_versions, args.pandas_range
    )
    print(",".join(versions))
