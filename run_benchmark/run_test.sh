#!/usr/bin/env bash

export KVIKIO_COMPAT_MODE=off

export LIBCUDF_CUFILE_POLICY="KVIKIO"
export LIBCUDF_LOGGING_LEVEL=INFO

export TMPDIR=/home/coder/cudf/run_benchmark

export CUFILE_ALLOW_COMPAT_MODE=false
export CUFILE_LOGGING_LEVEL=WARN

ctest --test-dir ~/cudf/cpp/build/latest -V -R PARQUET_TEST
