# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import http
import json
import ssl
import sys
import time
import urllib.error
import urllib.request

import certifi
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def has_python_version(version, data):
    for d in data:
        try:
            if Version(d["python_version"][2:]) == version:
                return True
        except Exception:
            pass
    return False


def get_pandas_versions(pandas_range):
    url = "https://pypi.org/pypi/pandas/json"
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    python_version = Version(
        str(sys.version_info.major) + str(sys.version_info.minor)
    )
    # Set a timeout for the request to avoid hanging
    timeout = 10  # seconds
    max_attempts = 3

    # Try to fetch pandas versions from PyPI

    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(
                url, timeout=timeout, context=ssl_context
            ) as response:
                data = json.loads(response.read())
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            if isinstance(e, urllib.error.HTTPError):
                code = e.code
            else:
                code = None

            if attempt == max_attempts - 1:
                raise e
            # Just retry retryable errors
            if (
                code
                in {
                    http.HTTPStatus.REQUEST_TIMEOUT,
                    http.HTTPStatus.TOO_MANY_REQUESTS,
                }
                or e.code >= 500
            ):
                print(f"HTTP error. Code={e.code}, attempt={attempt}")
                time.sleep(2**attempt)
                continue
            elif code is not None:
                print(
                    f"Non-retryable HTTP error. Code={e.code}, attempt={attempt}"
                )
                raise e
            else:
                # Assume it's retryable
                print(f"Retryable error. Code={code}, attempt={attempt}")
                time.sleep(2**attempt)
                continue

    # Extract and filter versions
    versions = [Version(v) for v in data["releases"]]

    specifier = SpecifierSet(pandas_range.lstrip("pandas"))
    matching_versions = [
        v
        for v in versions
        if v in specifier
        and has_python_version(python_version, data["releases"][str(v)])
    ]

    matching_minors = sorted(
        set(".".join((str(v.major), str(v.minor))) for v in matching_versions),
        key=Version,
    )

    return matching_minors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter pandas versions by prefix."
    )
    parser.add_argument(
        "pandas_range", type=str, help="The version prefix to filter by."
    )
    args = parser.parse_args()

    versions = get_pandas_versions(args.pandas_range)
    print(",".join(versions))
