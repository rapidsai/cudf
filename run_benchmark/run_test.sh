#!/usr/bin/env bash

export LIBCUDF_CUFILE_POLICY="GDS"
export CUFILE_ALLOW_COMPAT_MODE=false

# export CUFILE_ALLOW_COMPAT_MODE=false

export LIBCUDF_LOGGING_LEVEL=INFO

ctest --test-dir ~/cudf/cpp/build/latest -V -R PARQUET_TEST