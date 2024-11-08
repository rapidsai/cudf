# Copyright (c) 2024, NVIDIA CORPORATION.

import requests
from packaging.version import Version
from packaging.specifiers import SpecifierSet
import argparse

def get_pandas_versions(pandas_range):
    url = "https://pypi.org/pypi/pandas/json"
    response = requests.get(url)
    data = response.json()
    versions = [Version(v) for v in data['releases']]
    specifier = SpecifierSet(pandas_range.lstrip("pandas"))
    matching_versions = [v for v in versions if v in specifier]
    matching_minors = sorted(set(".".join((str(v.major), str(v.minor))) for v in matching_versions), key=Version)
    return matching_minors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter pandas versions by prefix.")
    parser.add_argument("pandas_range", type=str, help="The version prefix to filter by.")
    args = parser.parse_args()

    versions = get_pandas_versions(args.pandas_range)
    print(','.join(versions))
