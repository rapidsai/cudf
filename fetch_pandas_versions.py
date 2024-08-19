# Copyright (c) 2018-2024, NVIDIA CORPORATION.
import requests

def get_pandas_versions():
    url = "https://pypi.org/pypi/pandas/json"
    response = requests.get(url)
    data = response.json()
    versions = data['releases'].keys()

    # Filter out pre-releases and major versions
    minor_versions = sorted(set(
        '.'.join(version.split('.')[:2]) for version in versions if 'rc' not in version and 'b' not in version
    ))

    # Remove duplicates and sort
    minor_versions = sorted(set(minor_versions))

    # Filter for versions starting from 2.0
    minor_versions = [v for v in minor_versions if v.startswith('2.')]

    return minor_versions

if __name__ == "__main__":
    versions = get_pandas_versions()
    print(versions)
