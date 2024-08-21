# Copyright (c) 2024, NVIDIA CORPORATION.

import requests
from packaging.version import Version
from packaging.specifiers import SpecifierSet
import argparse

def get_pandas_versions(pandas_range):
    url = "https://pypi.org/pypi/pandas/json"
    response = requests.get(url)
    data = response.json()
    versions = data['releases'].keys()

    specifier = SpecifierSet(pandas_range)

    minor_versions = list(set([version[:3] for version in versions if Version(version) in specifier]))

    return minor_versions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter pandas versions by prefix.")
    parser.add_argument("pandas_range", type=str, help="The version prefix to filter by.")
    args = parser.parse_args()

    versions = get_pandas_versions(args.pandas_range)
    print(versions)
