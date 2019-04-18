# Copyright (c) 2018-2019, NVIDIA CORPORATION.

# Cleanup local git
git clean -xdf
# Change directory for build process
cd cpp/python
# build and install
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
