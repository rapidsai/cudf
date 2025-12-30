# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import http
import json
import ssl
import time
import urllib.error
import urllib.request

import certifi
import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def get_polars_specifier(deps_yaml_path):
    with open(deps_yaml_path, "r") as f:
        deps = yaml.safe_load(f)

    try:
        includes = deps["files"]["all"]["includes"]
        if "run_cudf_polars" not in includes:
            raise KeyError()
    except KeyError:
        raise RuntimeError("run_cudf_polars not found in dependencies.yaml")

    try:
        pkgs = deps["dependencies"]["run_cudf_polars"]["common"]
        for entry in pkgs:
            for pkg in entry.get("packages", []):
                if isinstance(pkg, str) and pkg.startswith("polars"):
                    spec = pkg.removeprefix("polars").strip()
                    if spec:
                        return spec
    except KeyError:
        pass

    raise RuntimeError("Polars specifier not found in dependencies.yaml")


def get_latest_versions_per_minor(versions):
    latest = {}
    for v in versions:
        key = (v.major, v.minor)
        if key not in latest or v > latest[key]:
            latest[key] = v
    return sorted(latest.values())


def get_polars_versions(polars_range, latest_only=False):
    url = "https://pypi.org/pypi/polars/json"
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    # Set a timeout for the request to avoid hanging
    timeout = 10  # seconds
    max_attempts = 3

    # Try to fetch polars versions from PyPI

    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(
                url, timeout=timeout, context=ssl_context
            ) as response:
                data = json.loads(response.read())
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
        ) as e:
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

    all_versions = [Version(v) for v in data["releases"]]
    specifier = SpecifierSet(polars_range)
    matching = [v for v in all_versions if v in specifier]

    if latest_only:
        matching = get_latest_versions_per_minor(matching)

    return [str(v) for v in sorted(matching)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter polars versions by dependencies.yaml."
    )
    parser.add_argument(
        "deps_yaml",
        nargs="?",
        default="./dependencies.yaml",
        help="Path to dependencies.yaml",
    )
    parser.add_argument(
        "--latest-patch-only",
        action="store_true",
        help="Return only the latest patch per minor version",
    )
    args = parser.parse_args()

    polars_range = get_polars_specifier(args.deps_yaml)
    versions = get_polars_versions(
        polars_range, latest_only=args.latest_patch_only
    )
    print(" ".join(versions))
