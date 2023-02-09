#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version>

VERSION=${1}

sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/cudf/cudf/__init__.py
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/dask_cudf/dask_cudf/__init__.py
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/cudf_kafka/cudf_kafka/__init__.py
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/custreamz/custreamz/__init__.py
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/strings_udf/strings_udf/__init__.py

sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/cudf/setup.py
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/dask_cudf/setup.py
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/cudf_kafka/setup.py
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/custreamz/setup.py
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/strings_udf/setup.py
